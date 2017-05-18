import tensorflow as tf
from tf_layer import *
import time 
import math
from IPython.display import display, HTML
import pandas as pd

class TFModel(object):
    def __init__(self, config, sess):
        self.sess = sess
        
        self.learning_rate = config.learning_rate #learning rate
        self.batch_size = config.batch_size #batch size to use during training
        self.clip_gradients = config.clip_gradients #max clip gradients
        
        self.n_epochs = config.n_epochs  #number of epoch to use during training
        self.epoch_save = config.epoch_save #save checkpoint or not in each epoch
        self.print_step = config.print_step #print step duraing training
        self.logs_dir = config.logs_dir
        self.model_dir = config.model_dir
        self.current_task_name = 'tf_models'
        #self.current_task_name = config.current_task_name
        if not os.path.isdir(self.logs_dir):
            os.makedirs(self.logs_dir)
            #raise Exception(" [!] Directory %s not found" % self.logs_dir)
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
            #raise Exception(" [!] Directory %s not found" % self.model_dir)
            
        self.dir_clear = config.dir_clear
        if self.dir_clear:
            model_logger_dir_prepare(self.logs_dir, self.model_dir, self.current_task_name)
        self.train_writer = tf.summary.FileWriter(self.logs_dir + self.current_task_name + '/train',
                                        self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.logs_dir + self.current_task_name + '/test')
        self.step_train = 0
        self.step_test = 0
        self.summary_step = 0
        self.saver = None
        self.metric_name = 'accuracy'
        self.gpu_option = config.gpu_id
        self.gpu_num = config.gpu_num
        #self.gpus = range(config.gpu_num) if config.gpu_num > 1 else [config.gpu_id]
        self.gpus = range(config.gpu_id, config.gpu_id+config.gpu_num)
        
        if hasattr(config,'opt'):
            self.opt = config.opt.lower()
        else:
            self.opt = 'adam'
            
        self.lr_annealing = config.lr_annealing
        self.lr_annealing_value = 1.5
        self.lr_stop_value = 1e-5
        
        #for summary
        self.loss_list = []
        self.optim_list = []  
        self.grad_list = []
        self.capped_grad_list = []
        self.metrics_list = []
        self.log_loss = []
        #for multiple gpus
        self.tower_grads = {}
        self.tower_capped_gvs = {}
        self.tower_loss = {}
        self.tower_metrics = {}
        self.tower_prediction_results = []
        self.params = None
        self.prediction_results = None
        self.lr = None
        self.fetch_dict = {}        
        tf.GraphKeys.INPUTS = 'inputs'
        
    def __add_details__(self, loss, optim, grads, capped_gvs, metric):
        self.loss_list.append(loss)
        self.optim_list.append(optim)
        self.grad_list.append(grads)
        self.capped_grad_list.append(capped_gvs)
        self.metrics_list.append(metric)
        
    def __better__(self, current_metric, best_metric):
        return current_metric >= best_metric
    
    def __worst_metric__(self):
        return 0.0
    
    def __set_metric_name__(self, name):
        self.metric_name = name
    
    def __get_feed_dict__(self, input_list, input_var, batch_idx, mode):
        feedDict = {}
        for j in range(len(input_list)):
            feedDict[input_var[j]] = input_list[j][batch_idx]
        return feedDict
    
    def __get_parameters_num__(self, shape_list):
        num = 1
        for l in shape_list:
            num *= l
        return num
    
    def __add_to_graph_input__(self, input_list):
        for each_input in input_list:
            tf.add_to_collection(tf.GraphKeys.INPUTS, each_input)
    
    def __build_global_setting__(self):
        with tf.name_scope('global'):
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            self.lr = tf.Variable(self.learning_rate)
            if self.opt == 'sgd':
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif self.opt == 'rmsprop':
                self.opt = tf.train.RMSPropOptimizer(self.lr, momentum=0.9)
            else:
                self.opt = tf.train.AdamOptimizer(self.lr)
    
    def __init_tower_list__(self, type):
        self.tower_grads[type] = []
        self.tower_capped_gvs[type] = []
        self.tower_loss[type] = []
        self.tower_metrics[type] = []
               
    def __add_to_tower_list__(self, grads, capped_gvs, loss, metric, type='self'):
        if type not in self.tower_grads:
            self.__init_tower_list__(type)
        self.tower_grads[type].append(grads)
        self.tower_capped_gvs[type].append(capped_gvs)
        self.tower_loss[type].append(loss)
        self.tower_metrics[type].append(metric)
    
    #aggregate the gradient, loass, metrics
    def build_model_aggregation(self):
        for key in self.tower_grads.keys():
            grads_avg = average_gradients(self.tower_grads[key])
            capped_gvs_avg = average_gradients(self.tower_capped_gvs[key])
            if len(self.tower_loss[key][0].shape) == 0:
                loss = tf.reduce_mean(self.tower_loss[key])
            else:
                loss = tf.concat(self.tower_loss[key], 0)
            if len(self.tower_metrics[key][0].shape) == 0:
                metric = tf.reduce_mean(self.tower_metrics[key])
            else:
                metric = tf.concat(self.tower_metrics[key], 0)
            train_op = self.opt.apply_gradients(capped_gvs_avg, global_step=self.global_step)
            self.__add_details__(loss, train_op, grads_avg, capped_gvs_avg, metric)  
            self.fetch_dict['{}_prediction'.format(key)] = [train_op, loss, metric]
        if len(self.tower_prediction_results) != 0:
            self.prediction_results = []
            for i in range(0, int(len(self.tower_prediction_results)/self.gpu_num)):
                temp = [self.tower_prediction_results[i] for i in range(i, len(self.tower_prediction_results), int(len(self.tower_prediction_results)/self.gpu_num))]
                self.prediction_results.append(tf.concat(temp, 0))
            if len(self.prediction_results) == 1:
                self.prediction_results = self.prediction_results[0]
        
            
    def build_model_summary(self, summary_step=0):
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name.replace(':', '_'), var)
        for loss in self.loss_list:
            tf.summary.scalar(loss.name.replace(':', '_'), loss)
        for metric in self.metrics_list:
            tf.summary.scalar(metric.name.replace(':', '_'), metric)
        for grads in self.grad_list:
            for grad, var in grads:
                tf.summary.histogram(var.name.replace(':', '_') + '/gradient', grad)
        for capped_gvs in self.capped_grad_list:
            for grad, var in capped_gvs:
                tf.summary.histogram(var.name.replace(':', '_') + '/clip_gradient', grad)
        self.merged = tf.summary.merge_all()
        for key in self.fetch_dict.keys():
            self.fetch_dict[key].append(self.merged)
        print('Initializing')
        self.saver = tf.train.Saver()
        self.summary_step = summary_step
        #with self.sess.graph.as_default():
        #self.sess.run(tf.global_variables_initializer())
        #self.sess.run(tf.local_variables_initializer())
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
    
    def model_summary(self):
        columns = ['variable_name','variable_shape', 'parameters']
        df_summary = pd.DataFrame(columns=columns)
        for i, var in enumerate(tf.trainable_variables()):
            sl = var.get_shape().as_list()
            df_summary.loc[i] = [var.name, sl, self.__get_parameters_num__(sl)]
        return df_summary
    
    def model_restore(self, model_path=None):
        if model_path == None:
            self.saver.restore(self.sess, '{}{}.ckpt'.format(self.model_dir,self.current_task_name))
        else:
            self.saver.restore(self.sess, model_path)
        print('Load Model ...')
              
    #training and evaluation    
    def model_run(self, input_list, idxs, run_type, mode='train', shuffle=True, save_metric=False):
        if len(idxs)%self.gpu_num!=0:
            idxs = idxs[:-(len(idxs)%self.gpu_num)]
        id_idxs = np.arange(0, len(idxs))
        if shuffle:
            np.random.shuffle(id_idxs)
        batchMetrics = []
        batchLoss = []
        #samplePreds = np.empty(shape=(len(idxs)))
        samplePreds = []
        input_var = tf.get_collection(tf.GraphKeys.INPUTS)
        for i, idx in enumerate(range(0, len(idxs), self.batch_size)):
            batch_idx = np.sort(idxs[id_idxs[idx:idx+self.batch_size]]).tolist()
            #batch_idx = idxs[id_idxs[idx:idx+self.batch_size]]
            feedDict = self.__get_feed_dict__(input_list, input_var, batch_idx, mode)
            #feedDict = {}
            #for j in range(len(input_list)):
            #    feedDict[input_var[j]] = input_list[j][batch_idx]
            if mode == 'train':
                #print('model training...')
                if self.summary_step==0 or i%self.summary_step!=0:
                    _, l, metr = self.sess.run(self.fetch_dict['{}_prediction'.format(run_type)][0:3], feed_dict=feedDict)
                else:
                    _, l, metr, summary = self.sess.run(self.fetch_dict['{}_prediction'.format(run_type)], feed_dict=feedDict)
                    self.train_writer.add_summary(summary, self.step_train)
                    self.step_train += 1
                if i>0 and i%self.print_step == 0:
                    print('Minibatch', i, '/', 'loss:', l if isinstance(l, np.float32) else np.mean(l))
                    print('Minibatch', i, '/', '{}:'.format(self.metric_name), metr if isinstance(metr, np.float32) else np.mean(metr))
            else:  #test
                #print('model testing...')
                if self.summary_step==0 or i%self.summary_step!=0:
                    l, metr = self.sess.run(self.fetch_dict['{}_prediction'.format(run_type)][1:3], feed_dict=feedDict)
                else:
                    l, metr, summary = self.sess.run(self.fetch_dict['{}_prediction'.format(run_type)][1:], feed_dict=feedDict)
                    self.test_writer.add_summary(summary, self.step_test)
                    self.step_test += 1
                if i>0 and i%self.print_step == 0:
                    print('Minibatch', i, '/', 'loss:', l if isinstance(l, np.float32) else np.mean(l))
                    print('Minibatch', i, '/', '{}:'.format(self.metric_name), metr if isinstance(metr, np.float32) else np.mean(metr))
                if (not shuffle) and save_metric:
                    #print('get results...')
                    if self.prediction_results is not None:
                        preds = self.sess.run(self.prediction_results, feed_dict=feedDict)
                        samplePreds.append(preds)
                        #sampleMetrics[id_idxs[idx:idx+self.batch_size]] = preds
            batchMetrics.append(metr*len(batch_idx) if isinstance(metr, np.float32) else np.sum(metr))       
            batchLoss.append(l*len(batch_idx) if isinstance(l, np.float32) else np.sum(l))
            #if len(samplePreds) != 0:
            #    samplePreds = np.concatenate(samplePreds)
        batchMetrics = np.array(batchMetrics)
        epochAcc = np.array(batchMetrics).sum() / len(id_idxs)
        epochLoss = np.array(batchLoss).sum() / len(id_idxs)
        return epochLoss, epochAcc, samplePreds
            
            
    def run(self, input_list, train_idxs, test_idxs, run_type='self', shuffle=True):
        best_train_metric = self.__worst_metric__()
        best_test_metric = self.__worst_metric__()
        input_var = tf.get_collection(tf.GraphKeys.INPUTS)
        for epoch in range(self.n_epochs):       
            print('Epoch', epoch+1, '... training ...')
            t1 = time.time()
            epochLoss, epochMetric, _ = self.model_run(input_list, train_idxs, run_type, mode='train', shuffle=shuffle)
            t2 = time.time()
            print('epoch time:', (t2-t1)/60)
            print('Epoch', epoch+1, 'training {}:'.format(self.metric_name), epochMetric)
            if self.__better__(epochMetric, best_train_metric):
                best_train_metric = epochMetric 
 
                    
            print('Epoch', epoch+1, '... test ...')      
            epochTestLoss, epochTestMetric, _ = self.model_run(input_list, test_idxs, run_type,
                                                               mode='test', shuffle=False)
            print('Epoch', epoch+1, 'test {}:'.format(self.metric_name), epochTestMetric)
            if self.__better__(epochTestMetric, best_test_metric): 
                best_test_metric = epochTestMetric
                if self.epoch_save:
                    save_path = self.saver.save(self.sess, '{}{}.ckpt'.format(self.model_dir,self.current_task_name))
                    print("Model saved in file: %s" % save_path)
                    
            self.log_loss.append([epochLoss, epochTestLoss])
            state = {
                'loss': epochLoss,
                'valid_los': epochTestLoss,
                'best_{}'.format(self.metric_name): best_train_metric,
                'best_test_{}'.format(self.metric_name): best_test_metric,
                'epoch': epoch,
                'learning_rate': self.learning_rate,   
                'valid_perplexity': math.exp(epochTestLoss)
                }
            print(state)
            
            if self.lr_annealing:
                if len(self.log_loss) > 1 and self.log_loss[epoch][1] > self.log_loss[epoch-1][1] * 0.9999:
                    self.learning_rate = self.learning_rate / self.lr_annealing_value
                    self.lr.assign(self.learning_rate).eval()
            if self.lr_annealing and self.learning_rate < self.lr_stop_value: break
            