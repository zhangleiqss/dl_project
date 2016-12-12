import sys
sys.path.append('../tftools')

from __future__ import absolute_import,division,print_function
import numpy as np
import tensorflow as tf
from utils import *
from past.builtins import xrange
from tensorflow.python.ops import rnn_cell, rnn
from tensorflow.python.ops.rnn import bidirectional_rnn,bidirectional_dynamic_rnn,dynamic_rnn
import math
import time


class SequentialModel(object):
    def __init__(self, config, sess):
        self.sess = sess
        
        self.nb_words = config.nb_words #term number in input sequence
        self.maxlen = config.maxlen #the max length of input sequence
        self.num_layers = config.num_layers #the number of rnn layers
        self.embedding_size = config.embedding_size  #word embedding size
        self.hidden_size = config.hidden_size #rnn hidden size
        self.keep_prob = config.keep_prob #keep probability of drop out
        self.learning_rate = config.learning_rate #learning rate
        self.batch_size = config.batch_size #batch size to use during training
        self.clip_gradients = config.clip_gradients #max clip gradients
        
        self.n_epochs = config.n_epochs  #number of epoch to use during training
        self.epoch_save = config.epoch_save #save checkpoint or not in each epoch
        self.print_step = config.print_step #print step duraing training
        self.logs_dir = config.logs_dir
        self.model_dir = config.model_dir
        self.current_task_name = config.current_task_name
        if not os.path.isdir(self.logs_dir):
            raise Exception(" [!] Directory %s not found" % self.logs_dir)
        if not os.path.isdir(self.model_dir):
            raise Exception(" [!] Directory %s not found" % self.model_dir)
            
        self.dir_clear = config.dir_clear
        if self.dir_clear:
            model_logger_dir_prepare(self.logs_dir, self.model_dir, self.current_task_name)
        self.train_writer = tf.train.SummaryWriter(self.logs_dir + self.current_task_name + '/train',
                                        self.sess.graph)
        self.test_writer = tf.train.SummaryWriter(self.logs_dir + self.current_task_name + '/test')
        self.step_train = 0
        self.step_test = 0
        self.saver = None
        self.gpu_option = '/gpu:{}'.format(config.gpu_id)

        self.lr_annealing = config.lr_annealing
        self.lr_annealing_value = 1.5
        self.lr_stop_value = 1e-5
             
        #for self prediction
        self.input_params = None
        self.targets_list = []
        self.nb_class_list = []     
        self.loss_list = []
        self.optim_list = []
        self.params_list = []
        self.grad_list = []
        self.capped_grad_list = []
        self.metrics_list = []
        self.log_loss = []  
        self.logits_list = []
        self.fetch_dict = {}
        self.lr_dict = {} 
        self.pred_dict = {}
        tf.GraphKeys.INPUTS = 'inputs'
    
    #build the input RNN
    def build_input_sequence(self):       
        with tf.device(self.gpu_option): 
            with tf.name_scope('input'):
                self.inputX = tf.placeholder(tf.int32, shape=(None, self.maxlen))
                self.seqLengths = tf.placeholder(tf.int64, shape=(None))
            
                tf.add_to_collection(tf.GraphKeys.INPUTS,  self.inputX)
                tf.add_to_collection(tf.GraphKeys.INPUTS,  self.seqLengths)
            with tf.name_scope('rnn_input'):
                self.input_embedding = my_embedding_layer(self.inputX, self.nb_words, self.embedding_size)
            with tf.name_scope('rnn'):
                single_cell = rnn_cell.LSTMCell(self.hidden_size, use_peepholes=True, state_is_tuple=True)
                single_cell = rnn_cell.DropoutWrapper(single_cell, input_keep_prob=self.keep_prob, 
                                                  output_keep_prob=self.keep_prob)
                cell = rnn_cell.MultiRNNCell([single_cell] * self.num_layers, state_is_tuple=True)
                self.state, self.output = dynamic_rnn(cell, self.input_embedding, 
                                                  self.seqLengths, dtype=tf.float32)
                self.state_list = tf.unpack(self.state,axis=1)
                self.input_params = tf.trainable_variables()
        
    @deprecated
    def build_input_history_attention(self, context=5, beta=1, alpha=1):
        with tf.device(self.gpu_option): 
            input_embedding_list = tf.unpack(self.input_embedding, axis=1)         
            input_embedding_extend = tf.concat(1, [tf.zeros_like(self.input_embedding[:,0:context,:]),self.input_embedding])
            state_extend = tf.concat(1, [tf.zeros_like(self.state[:,0:context,:]),self.state])
            attention_state_list = []
            for i in range(self.maxlen):
                similarity = tf.batch_matmul(tf.reshape(input_embedding_list[i], [-1, 1, self.embedding_size]), tf.slice(input_embedding_extend, [0,i,0],[-1,context,-1]), adj_y=True)
                attention = tf.nn.softmax(similarity*beta)
                new_state = tf.batch_matmul(attention, tf.transpose(tf.slice(state_extend,[0,0,0],[-1,context,-1]),[0,2,1]), adj_y=True)
                attention_state_list.append(new_state)
            attention_state = tf.concat(1,attention_state_list)          
            self.state = self.state + attention_state*alpha
            self.state_list = tf.unpack(self.state,axis=1) 
                
    #build the prediction for each time step    
    def build_sequence_prediction(self, type='self', nb_class=None, accK=5):
        with tf.device(self.gpu_option):
            with tf.name_scope('{}_output'.format(type)):
                if type == 'self':
                    targets = tf.concat(1, [tf.slice(self.inputX,[0,1],[-1,-1]),
                                          tf.zeros_like(self.inputX[:,0:1])])
                else:
                    targets = tf.placeholder(tf.int32, shape=(None, self.maxlen))
                    tf.add_to_collection(tf.GraphKeys.INPUTS, targets)
                self.targets_list.append(targets)
            with tf.name_scope('{}_train'.format(type)):
                if type == 'self':
                    nb_class = self.nb_words
                elif nb_class == None:
                    raise Exception("nb_class must be given")
                self.nb_class_list.append(nb_class)
                logits = [my_full_connected(t,self.hidden_size, nb_class, reuse=(i!=0),
                          layer_name='{}_time_distributed_fc1'.format(type), 
                          act=tf.identity) for i,t in enumerate(self.state_list)]
                self.logits_list.append(logits)
            with tf.name_scope('{}_loss'.format(type)):
                
                loss = variable_seq_loss(tf.pack(logits,axis=1), targets)       
                lr = tf.Variable(self.learning_rate)           
                #will use rewrite by optimizer.py
                #opt = tf.train.RMSPropOptimizer(lr)
                opt = tf.train.AdamOptimizer(lr)
                params = [var for var in self.input_params] 
                #add variable W and b in logits, a little trick
                params.extend(tf.trainable_variables()[-3:-1])      
                optimizer, grads, capped_gvs = my_minimize_loss(opt, loss, params, clip_type= 'clip_global_norm', max_clip_grad=self.clip_gradients) 
                self.loss_list.append(loss)
                self.lr_dict['{}_prediction'.format(type)] = lr
                self.params_list.append(params)
                self.optim_list.append(optimizer)
                self.grad_list.append(grads)
                self.capped_grad_list.append(capped_gvs)
            with tf.name_scope('{}_accuracy'.format(type)):
                #accuracy = varibale_accuracy(tf.pack(logits,axis=1), targetsY)
                accuracy = varibale_topk_accuracy(tf.pack(logits,axis=1), targets, k=accK)    
                self.metrics_list.append(accuracy)
                self.pred_dict[type] = [tf.pack(logits,axis=1), targets]
            self.fetch_dict['{}_prediction'.format(type)] = [self.optim_list[-1], self.loss_list[-1],self.metrics_list[-1]]
    
    #build the prediction for last time step
    def build_single_prediction(self, type='single', nb_class=2, accK=1):
        with tf.device(self.gpu_option):
            with tf.name_scope('{}_output'.format(type)):
                targets = tf.placeholder(tf.int32, shape=(None))
                tf.add_to_collection(tf.GraphKeys.INPUTS, targets)
                self.targets_list.append(targets)
            with tf.name_scope('{}_train'.format(type)):
                self.nb_class_list.append(nb_class)
                prediction = my_full_connected(self.output[0][1], self.hidden_size, nb_class, 
                                           layer_name='{}_fc'.format(type),act=tf.identity)
            with tf.name_scope('{}_loss'.format(type)):
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(prediction, targets))
                lr = tf.Variable(self.learning_rate)           
                opt = tf.train.AdamOptimizer(lr)
                params = [var for var in self.input_params] 
                #add variable W and b in logits, a little trick
                params.extend(tf.trainable_variables()[-3:-1])      
                optimizer, grads, capped_gvs = my_minimize_loss(opt, loss, params, clip_type= 'clip_global_norm', max_clip_grad=self.clip_gradients)
                self.loss_list.append(loss)
                self.lr_dict['{}_prediction'.format(type)] = lr
                self.params_list.append(params)
                self.optim_list.append(optimizer)
                self.grad_list.append(grads)
                self.capped_grad_list.append(capped_gvs)
            with tf.name_scope('{}_accuracy'.format(type)):
                accuracy = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(prediction,targets,k=accK)))
                #accuracy = tf.to_float(tf.nn.in_top_k(prediction,targets,k=accK))
                self.metrics_list.append(accuracy)
                self.pred_dict[type] = [prediction, targets]
            self.fetch_dict['{}_prediction'.format(type)] = [self.optim_list[-1], self.loss_list[-1], self.metrics_list[-1]]
    
    def build_model_summary(self, summary_step=0):
        for var in tf.trainable_variables():
            tf.histogram_summary(var.name, var)
        for loss in self.loss_list:
            tf.scalar_summary(loss.name, loss)
        for metric in self.metrics_list:
            tf.scalar_summary(metric.name, metric)
        for grads in self.grad_list:
            for grad, var in grads:
                tf.histogram_summary(var.name + '/gradient', grad)
        for capped_gvs in self.capped_grad_list:
            for grad, var in capped_gvs:
                tf.histogram_summary(var.name + '/clip_gradient', grad)
        self.merged = tf.merge_all_summaries()
        for key in self.fetch_dict.keys():
            self.fetch_dict[key].append(self.merged)
        print('Initializing')
        self.saver = tf.train.Saver()
        self.summary_step = summary_step
        tf.initialize_all_variables().run()
   
    def model_restore(self, final=True, epoch=0):
        if final == True:
            self.saver.restore(self.sess, '{}{}/final.ckpt'.format(self.model_dir,self.current_task_name))
        else:
            self.saver.restore(self.sess, '{}{}/{}.ckpt'.format(self.model_dir,self.current_task_name,epoch))

    #get the sequence output
    def input_sequence_output_eval(self, input_list, idxs):
        idxs = np.sort(idxs)
        id_idxs = np.arange(0, len(idxs))
        input_var = tf.get_collection(tf.GraphKeys.INPUTS)[0:2]
        assert len(input_var) == len(input_list), 'the length of input list is not match'
        state_array = np.empty(shape=(len(idxs), self.maxlen, self.hidden_size))
        output_array = np.empty(shape=(len(idxs), self.hidden_size))
        for i, idx in enumerate(range(0, len(idxs), self.batch_size)):
            batch_idx = idxs[id_idxs[idx:idx+self.batch_size]].tolist()
            feedDict = {}
            for j in xrange(len(input_list)):
                feedDict[input_var[j]] = input_list[j][batch_idx]
            state, output = self.sess.run([self.state, self.output[0][1]], feed_dict=feedDict)
            state_array[id_idxs[idx:idx+self.batch_size]] = state
            output_array[id_idxs[idx:idx+self.batch_size]] = output
        return state_array, output_array         
                     
            
    #training and evaluation    
    def model_run(self, input_list, idxs, run_type, mode='train', shuffle=True):
        id_idxs = np.arange(0, len(idxs))
        if shuffle:
            np.random.shuffle(id_idxs)
        batchMetrics = []
        batchLoss = []
        input_var = tf.get_collection(tf.GraphKeys.INPUTS)
        for i, idx in enumerate(range(0, len(idxs), self.batch_size)):
            batch_idx = np.sort(idxs[id_idxs[idx:idx+self.batch_size]]).tolist()
            feedDict = {}
            for j in xrange(len(input_list)):
                feedDict[input_var[j]] = input_list[j][batch_idx]
            if mode == 'train':
                if self.summary_step==0 or i%self.summary_step!=0:
                    _, l, acc = self.sess.run(self.fetch_dict['{}_prediction'.format(run_type)][0:3], feed_dict=feedDict)
                else:
                     _, l, acc, summary = self.sess.run(self.fetch_dict['{}_prediction'.format(run_type)], feed_dict=feedDict)
                        self.train_writer.add_summary(summary, self.step_train)
                        self.step_train += 1
                if i%self.print_step == 0:
                    print('Minibatch', i, '/', 'loss:', l)
                    print('Minibatch', i, '/', 'acc:', acc)
            else:
                if self.summary_step==0 or i%self.summary_step!=0:
                    l, acc = self.sess.run(self.fetch_dict['{}_prediction'.format(run_type)][1:3], feed_dict=feedDict)
                else:
                    l, acc, summary = self.sess.run(self.fetch_dict['{}_prediction'.format(run_type)][1:], feed_dict=feedDict)
                    self.test_writer.add_summary(summary, self.step_test)
                    self.step_test += 1
            batchMetrics.append(acc*len(batch_idx))
            batchLoss.append(l*len(batch_idx))
        batchMetrics = np.array(batchMetrics)
        epochAcc = np.array(batchMetrics).sum() / len(id_idxs)
        epochLoss = np.array(batchLoss).sum() / len(id_idxs)
        return epochLoss, epochAcc
    
    def run(self, input_list, train_idxs, test_idxs, run_type='self'):
        best_train_acc = 0.0
        best_test_acc = 0.0
        input_var = tf.get_collection(tf.GraphKeys.INPUTS)
        assert len(input_var) == len(input_list), 'the length of input list is not match'
        for epoch in xrange(self.n_epochs):
            print('Epoch', epoch+1, '... training ...')
            epochLoss, epochAcc = self.model_run(input_list, train_idxs, run_type, mode='train')
            print('Epoch', epoch+1, 'training acc:', epochAcc)
            if best_train_acc < epochAcc:
                best_train_acc = epochAcc 
            
            if self.epoch_save:
                save_path = self.saver.save(self.sess, '{}{}/{}.ckpt'.format(self.model_dir,self.current_task_name,epoch))
                print("Model saved in file: %s" % save_path)
            
            #model testing...     
            print('Epoch', epoch+1, '... test ...')      
            epochTestLoss, epochTestAcc = self.model_run(input_list, test_idxs, run_type,
                                                         mode='test', shuffle=False)     
            if best_test_acc < epochTestAcc: 
                best_test_acc = epochTestAcc
            print('Epoch', epoch+1, 'test acc:', epochTestAcc)
    
            self.log_loss.append([epochLoss, epochTestLoss])
            state = {
                'loss': epochLoss,
                'valid_los': epochTestLoss,
                'best_accuracy': best_train_acc,
                'best_test_accuracy': best_test_acc,
                'epoch': epoch,
                'learning_rate': self.learning_rate,        
                }
            print(state)
        
        self.train_writer.close()
        self.test_writer.close()
        save_path = self.saver.save(self.sess, '{}{}/final.ckpt'.format(self.model_dir,self.current_task_name))
        print("Model saved in file: %s" % save_path)
