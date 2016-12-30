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
            raise Exception(" [!] Directory %s not found" % self.logs_dir)
        if not os.path.isdir(self.model_dir):
            raise Exception(" [!] Directory %s not found" % self.model_dir)
            
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
        #self.gpu_option = '/gpu:{}'.format(config.gpu_id)
        self.gpu_option = config.gpu_id
        self.gpu_num = config.gpu_num
        self.gpus = range(config.gpu_num) if config.gpu_num > 0 else [config.gpu_id]
        
        self.lr_annealing = config.lr_annealing
        self.lr_annealing_value = 1.5
        self.lr_stop_value = 1e-5
        
        self.loss_list = []
        self.params_list = []
        self.optim_list = []  
        self.grad_list = []
        self.capped_grad_list = []
        self.metrics_list = []
        self.log_loss = []
        self.tower_grads = []
        self.tower_capped_gvs = []
        self.tower_loss = []
        self.tower_metrics = []
        self.fetch_dict = {}
        self.lr = None
        tf.GraphKeys.INPUTS = 'inputs'
        
    def __add_details__(self, loss, params, optim, grads, capped_gvs, metric):
        self.loss_list.append(loss)
        self.params_list.append(params)
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
        num = 1L
        for l in shape_list:
            num *= l
        return num

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
        tf.global_variables_initializer().run()
    
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
        sampleMetrics = np.empty(shape=(len(idxs)))
        input_var = tf.get_collection(tf.GraphKeys.INPUTS)
        for i, idx in enumerate(range(0, len(idxs), self.batch_size)):
            batch_idx = np.sort(idxs[id_idxs[idx:idx+self.batch_size]]).tolist()
            feedDict = self.__get_feed_dict__(input_list, input_var, batch_idx, mode)
            #feedDict = {}
            #for j in range(len(input_list)):
            #    feedDict[input_var[j]] = input_list[j][batch_idx]
            if mode == 'train':
                if self.summary_step==0 or i%self.summary_step!=0:
                    _, l, metr = self.sess.run(self.fetch_dict['{}_prediction'.format(run_type)][0:3], feed_dict=feedDict)
                else:
                    _, l, metr, summary = self.sess.run(self.fetch_dict['{}_prediction'.format(run_type)], feed_dict=feedDict)
                    self.train_writer.add_summary(summary, self.step_train)
                    self.step_train += 1
                if i>0 and i%self.print_step == 0:
                    print('Minibatch', i, '/', 'loss:', l if isinstance(l, np.float32) else np.mean(l))
                    print('Minibatch', i, '/', '{}:'.format(self.metric_name), metr if isinstance(metr, np.float32) else np.mean(metr))
            else:
                if self.summary_step==0 or i%self.summary_step!=0:
                    l, metr = self.sess.run(self.fetch_dict['{}_prediction'.format(run_type)][1:3], feed_dict=feedDict)
                else:
                    l, metr, summary = self.sess.run(self.fetch_dict['{}_prediction'.format(run_type)][1:], feed_dict=feedDict)
                    self.test_writer.add_summary(summary, self.step_test)
                    self.step_test += 1
                if (not shuffle) and isinstance(metr, np.ndarray) and save_metric:
                    sampleMetrics[id_idxs[idx:idx+self.batch_size]] = metr
            batchMetrics.append(metr*len(batch_idx) if isinstance(metr, np.float32) else np.sum(metr))       
            batchLoss.append(l*len(batch_idx) if isinstance(l, np.float32) else np.sum(l))
        batchMetrics = np.array(batchMetrics)
        epochAcc = np.array(batchMetrics).sum() / len(id_idxs)
        epochLoss = np.array(batchLoss).sum() / len(id_idxs)
        return epochLoss, epochAcc, sampleMetrics
            
            
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
            if self.lr_annealing and self.lr_stop_value < 1e-5: break
            