from __future__ import absolute_import,division,print_function
import sys
sys.path.append('../tftools')

import numpy as np
import tensorflow as tf
from tf_object import *
from tensorflow.python.ops import rnn_cell, rnn
from tensorflow.python.ops.rnn import bidirectional_rnn,bidirectional_dynamic_rnn,dynamic_rnn



class SequentialModel(TFModel):
    def __init__(self, config, sess):
        super(SequentialModel, self).__init__(config, sess)
        
        self.nb_words = config.nb_words #term number in input sequence
        self.maxlen = config.maxlen #the max length of input sequence
        self.num_layers = config.num_layers #the number of rnn layers
        self.embedding_size = config.embedding_size  #word embedding size
        self.hidden_size = config.hidden_size #rnn hidden size
        self.keep_prob = config.keep_prob #keep probability of drop out
                  
        #for self prediction
        self.input_params = None
        self.targets_list = []
        self.nb_class_list = []             
        self.logits_list = []
        self.lr_dict = {} 
        self.pred_dict = {}
    
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
        
    #deprecated
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
                self.lr_dict['{}_prediction'.format(type)] = lr
            with tf.name_scope('{}_accuracy'.format(type)):
                #accuracy = varibale_accuracy(tf.pack(logits,axis=1), targetsY)
                accuracy = varibale_topk_accuracy(tf.pack(logits,axis=1), targets, k=accK)    
                self.pred_dict[type] = [tf.pack(logits,axis=1), targets]
                self.__add_details__(loss, params, optimizer, grads, capped_gvs, accuracy)
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
                self.lr_dict['{}_prediction'.format(type)] = lr
            with tf.name_scope('{}_accuracy'.format(type)):
                accuracy = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(prediction,targets,k=accK)))
                #accuracy = tf.to_float(tf.nn.in_top_k(prediction,targets,k=accK))
                self.pred_dict[type] = [prediction, targets]
                self.__add_details__(loss, params, optimizer, grads, capped_gvs, accuracy)
            self.fetch_dict['{}_prediction'.format(type)] = [self.optim_list[-1], self.loss_list[-1], self.metrics_list[-1]]
      
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
                     
            
 