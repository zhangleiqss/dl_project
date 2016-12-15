import sys
sys.path.append('../tftools')

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import ctc_ops as ctc
from tf_object import *
import time
import ast
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn,dynamic_rnn

class ASRBaseModel(TFModel):
    def __init__(self, config, sess):
        super(ASRBaseModel, self).__init__(config, sess)
        self.mfcc_features = config.mfcc_features
        self.voca_size = config.voca_size
        self.logits = None
        self.optimizer = 'sgd'  
        self.current_task_name = 'asrmodel'
        self.metric_name = 'error_rate'
        #self.fetch_dict = None
    def __better__(self, current_metric, best_metric):
        return current_metric <= best_metric
    
    def __worst_metric__(self):
        return 1.0
    
    def __get_feed_dict__(self, input_list, input_var, batch_idx, mode):
        feedDict = {}
        temp_x = input_list[0][batch_idx] #inputX   
        feedDict[input_var[0]] = np.zeros(shape=(len(batch_idx),np.max(input_list[1][batch_idx]), self.mfcc_features))
        for j in range(len(batch_idx)):
            feedDict[input_var[0]][j][:input_list[1][batch_idx][j],:] = np.transpose(temp_x[j])
        feedDict[input_var[1]] = input_list[1][batch_idx] #seqLengths
        feedDict[input_var[2]], feedDict[input_var[3]], feedDict[input_var[4]] = target_list_to_sparse_tensor(input_list[2][batch_idx])
        feedDict[input_var[5]] = (mode=='train')       
        return feedDict    
    def build_input_graph(self):       
        with tf.device(self.gpu_option): 
            with tf.name_scope('input'):
                self.inputX = tf.placeholder(tf.float32, shape=(None, None, self.mfcc_features))
                self.seqLengths = tf.placeholder(tf.int32, shape=(None))
                targetIxs = tf.placeholder(tf.int64)
                targetVals = tf.placeholder(tf.int32)
                targetShape = tf.placeholder(tf.int64)
                self.targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
                self.training = tf.placeholder(tf.bool)
                tf.add_to_collection(tf.GraphKeys.INPUTS,  self.inputX)
                tf.add_to_collection(tf.GraphKeys.INPUTS,  self.seqLengths)
                tf.add_to_collection(tf.GraphKeys.INPUTS,  targetIxs)
                tf.add_to_collection(tf.GraphKeys.INPUTS,  targetVals)
                tf.add_to_collection(tf.GraphKeys.INPUTS,  targetShape)
                tf.add_to_collection(tf.GraphKeys.INPUTS,  self.training)
    
    def build_model(self):
        with tf.device(self.gpu_option): 
            with tf.name_scope('model'):
                self.logits = my_conv_1d(self.inputX, 1, self.voca_size, add_bias=False, act=tf.identity)
                
    def build_model_loss(self, optimizer='adam', clip_type='clip_norm'):
        with tf.device(self.gpu_option): 
            with tf.name_scope('loss'):
                self.loss = ctc.ctc_loss(self.logits, self.targetY, self.seqLengths, time_major=False)
                predictions = tf.to_int32(ctc.ctc_beam_search_decoder(tf.transpose(self.logits,[1,0,2]), self.seqLengths, merge_repeated=False)[0][0])
                self.errorRate = tf.reduce_sum(tf.edit_distance(predictions, self.targetY, normalize=False)) / tf.to_float(tf.size(self.targetY.values))
                self.predictions_dense = tf.sparse_to_dense(predictions.indices, predictions.shape, predictions.values) + 1
            with tf.name_scope('training'):
                self.params = tf.trainable_variables()
                self.lr = tf.Variable(self.learning_rate)
                if optimizer == 'adam':
                    optim =  tf.train.AdamOptimizer(self.lr)
                    self.optimizer = 'adam'
                elif optimizer == 'rmsprop':
                    optim =  tf.train.RMSPropOptimizer(self.lr, momentum=0.9)
                    self.optimizer = 'rmsprop'
                else:
                    optim = tf.train.GradientDescentOptimizer(self.lr)
                self.opt, grads, capped_gvs = my_minimize_loss(optim, self.loss, self.params, 
                                                           clip_type=clip_type, max_clip_grad=self.clip_gradients)
            self.fetch_dict['self_prediction'] = [self.opt, self.loss, self.errorRate]
            #self.saver = tf.train.Saver()  
            #tf.initialize_all_variables().run() 
        #for var in self.params:
        #    print(var.name, var.get_shape())
        
    def print_model_summary(self):
        for var in self.params:
            print(var.name, var.get_shape())
            
    
              
class WaveNet(ASRBaseModel):
    def __init__(self, config, sess, current_task_name='wavenet'):
        super(WaveNet, self).__init__(config, sess)
        self.conv_dim = config.conv_dim
        self.rate = config.rate
        self.num_blocks = config.num_blocks
        self.current_task_name = current_task_name
        
    def res_block(self, input_tensor, conv_length, rate, n_filters_out, training):
        conv_filter = my_atrous_conv_1d(input_tensor, conv_length, n_filters_out, rate, add_bias=False)
        conv_filter_bn = my_batch_norm(conv_filter, training)
        conv_gate = my_atrous_conv_1d(input_tensor, conv_length, n_filters_out, rate, add_bias=False, act=tf.nn.sigmoid)
        conv_gate_bn = my_batch_norm(conv_gate, training)
        out = conv_filter_bn * conv_gate_bn
        out_conv = my_conv_1d(out,1,n_filters_out,add_bias=False)
        out_conv_bn = my_batch_norm(out_conv, training)
        return out_conv_bn + input_tensor, out_conv_bn

    def build_model(self, keep_prob=1):
        with tf.device(self.gpu_option): 
            with tf.name_scope('model'):
                with tf.name_scope('fc_conv'):
                    z = my_conv_1d(self.inputX, 1, self.conv_dim, add_bias=False)
                    z = my_batch_norm(z, self.training)
                with tf.name_scope('skip_atrous_conv_blocks'):
                    skip = 0  # skip connections
                    for i in range(self.num_blocks):
                        for r in [1, 2, 4, 8, 16]:
                            z, s = self.res_block(z, self.rate, r, self.conv_dim, self.training)
                            skip += s
                with tf.name_scope('fc'):
                    if keep_prob > 0 and keep_prob < 1:
                        skip = tf.nn.dropout(skip, keep_prob)
                    logit = my_conv_1d(skip, 1, self.conv_dim, add_bias=False)
                    logit = my_batch_norm(logit, self.training)
                    self.logits = my_conv_1d(logit, 1, self.voca_size, add_bias=False, act=tf.identity)
                    #print(self.logits)

class DeepSpeech(ASRBaseModel):
    def __init__(self, config, sess, current_task_name='deepspeech'):
        super(DeepSpeech, self).__init__(config, sess)
        self.fc_conv_dim = config.fc_conv_dim   #the 1-conv dim
        self.conv_length_list = ast.literal_eval(config.conv_length_list)  #tuple or list
        self.conv_n_filters_out_list = ast.literal_eval(config.conv_n_filters_out_list)
        self.rnn_dim = config.rnn_dim
        self.rnn_num_layers = config.rnn_num_layers
        self.current_task_name = current_task_name
        
    def build_model(self, conv_keep_prob=1, rnn_keep_prob=1, zoneout_state=0.0, zoneout_cell=0.0):
        with tf.device(self.gpu_option): 
            with tf.name_scope('model'):
                with tf.name_scope('fc_conv'):
                    conv_x = my_conv_1d(self.inputX, 1, self.fc_conv_dim, add_bias=False)
                    conv_x = my_batch_norm(conv_x, self.training)
                with tf.name_scope('conv'):
                    for i in range(len(self.conv_length_list)):
                        conv_x = tf.pad(conv_x, [[0, 0], [(self.conv_length_list[i]-1), 0], [0, 0]])
                        conv_x = my_conv_1d(conv_x, self.conv_length_list[i], self.conv_n_filters_out_list[i], padding='VALID', add_bias=False)
                        conv_x = my_batch_norm(conv_x, self.training)
                    if conv_keep_prob > 0 and conv_keep_prob < 1:
                        conv_x = tf.nn.dropout(conv_x, conv_keep_prob)
                with tf.name_scope('rnn'):
                    single_cell = rnn_cell.LSTMCell(self.rnn_dim, use_peepholes=True, state_is_tuple=True)
                    #rnn dropout
                    if zoneout_state + zoneout_cell > 1e-7:
                        single_cell = ZoneoutWrapper(single_cell, self.training, zoneout_state, zoneout_cell)
                    elif rnn_keep_prob > 0 and rnn_keep_prob < 1:
                        single_cell = rnn_cell.DropoutWrapper(single_cell, input_keep_prob=rnn_keep_prob, output_keep_prob=rnn_keep_prob)                 
                    cell = rnn_cell.MultiRNNCell([single_cell] * self.rnn_num_layers, state_is_tuple=True)    
                    state, _  = bidirectional_dynamic_rnn(cell, cell, conv_x, 
                                                         sequence_length=tf.to_int64(self.seqLengths),
                                                         dtype=tf.float32)
                    bi_state = tf.concat(2, [state[0],state[1]])
                with tf.name_scope('fc'):        
                    self.logits = my_conv_1d(bi_state, 1, self.voca_size, add_bias=False, act=tf.identity)
                    
    def build_model_loss(self, optimizer='rmsprop', clip_type='clip_norm'):
        #print(optimizer)
        super(DeepSpeech, self).build_model_loss(optimizer, clip_type)
        
class DeepConvAcousticModel(ASRBaseModel):
    def __init__(self, config, sess, current_task_name='deepconv'):
        super(DeepConvAcousticModel, self).__init__(config, sess)
        self.first_conv_dim = config.first_conv_dim
        self.first_conv_length = config.first_conv_length
        self.block_conv_length = config.block_conv_length
        self.block_conv_dim = config.block_conv_dim
        self.block_nums = config.block_nums
        self.last_conv_dim = config.last_conv_dim
        self.last_conv_length = config.last_conv_length
        self.current_task_name = current_task_name
        
    def build_model(self, keep_prob=1):
        with tf.device(self.gpu_option): 
            with tf.name_scope('model'):
                with tf.name_scope('fc_conv'):
                    conv_x = tf.pad(self.inputX, [[0, 0], [(self.first_conv_length-1), 0], [0, 0]])
                    conv_x = my_conv_1d(conv_x, self.first_conv_length, self.first_conv_dim, add_bias=False, stride_step=1,padding='VALID')
                    conv_x = my_batch_norm(conv_x, self.training)
                    #self.seqLengths = (self.seqLengths+1)/2
                with tf.name_scope('conv_block'):
                    for i in range(self.block_nums):
                        conv_x = tf.pad(conv_x, [[0, 0], [(self.block_conv_length-1), 0], [0, 0]])
                        conv_x = my_conv_1d(conv_x, self.block_conv_length, self.block_conv_dim, padding='VALID', add_bias=False)
                        conv_x = my_batch_norm(conv_x, self.training)
                    if keep_prob > 0 and keep_prob < 1:
                        conv_x = tf.nn.dropout(conv_x,keep_prob)
                with tf.name_scope('last_conv'):
                    conv_x = tf.pad(conv_x, [[0, 0], [(self.last_conv_length-1), 0], [0, 0]])
                    conv_x = my_conv_1d(conv_x, self.last_conv_length, self.last_conv_dim, padding='VALID', add_bias=False)
                    conv_x = my_batch_norm(conv_x, self.training)
                    conv_x = my_conv_1d(conv_x, 1, self.last_conv_dim, add_bias=False)
                    conv_x = my_batch_norm(conv_x, self.training)
                with tf.name_scope('fc'):
                    self.logits = conv_x = my_conv_1d(conv_x, 1, self.voca_size , add_bias=False, act=tf.identity)
                                     