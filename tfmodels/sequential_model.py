from __future__ import absolute_import,division,print_function
import sys
sys.path.append('../tfmodels')

import numpy as np
import tensorflow as tf
from rnnlm import *
#from tensorflow.python.ops import rnn_cell, rnn
#from tensorflow.python.ops.rnn import bidirectional_rnn,bidirectional_dynamic_rnn,dynamic_rnn


class SequentialModel(RNNLMModel):
    def __init__(self, config, sess, current_task_name='sequence_model'):
        super(SequentialModel, self).__init__(config, sess, current_task_name)
    
    def build_single_prediction(self, gpu_id=0, accK=5, nb_class=None):
        self.params_1 = None
        if nb_class is None:
            nb_class = self.nb_words
        with get_new_variable_scope('prediction') as pred_scope:    
            prediction = my_full_connected(self.output_list[gpu_id][0][-1], nb_class, 
                                       add_bias=True, act=tf.identity, init_std=self.init_std)
            self.tower_prediction_results.append(tf.nn.softmax(prediction))
        with tf.name_scope('loss'): 
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.split_label[gpu_id], 
                                                                  logits=prediction)

            self.params_1 = [param for param in self.input_params]
            self.params_1.extend(tf.trainable_variables()[-2:])
            grads, capped_gvs = my_compute_grad(self.opt, loss, self.params_1, 
                                                clip_type = 'clip_norm', 
                                                max_clip_grad=self.clip_gradients)            
        with tf.name_scope('accuracy'):
            accuracy = tf.to_float(tf.nn.in_top_k(prediction, self.split_label[gpu_id],k=accK))        
        self.__add_to_tower_list__(grads, capped_gvs, loss, accuracy, 'single')
    
    def build_single_output(self):
        with tf.name_scope('output'):
            label = tf.placeholder(tf.int64, [None], name="label")
            self.__add_to_graph_input__([label])
            self.split_label = tf.split(label, self.gpu_num, 0)
    
    def build_output(self, type='self'):
        if isinstance(type, list):
            super(SequentialModel, self).build_output(type[0])
            self.build_single_output()
        else:
            if type == 'single':
                self.build_single_output()
            else:
                super(SequentialModel, self).build_output(type)
    
    def split_parameter(self, param):
        if isinstance(param, list):
            if len(param) > 1:
                return param[0], param[1]
            else:
                return param[0], param[0]
        else:
            return param, param
    
    
    def build_model(self, type=['self','single'], accK=5, nb_class=None):
        self.build_input()
        self.build_output(type)
        accK1, accK2 = self.split_parameter(accK)
        nb_class1, nb_class2 = self.split_parameter(nb_class)
        new_type = type[0] if isinstance(type,list) else type
        for idx, gpu_id in enumerate(self.gpus):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('Tower_%d' % (gpu_id)) as tower_scope:
                    gpu_scope = tf.variable_scope('gpu', reuse=(idx!=0))
                    with gpu_scope as gpu_scope:
                        self.build_input_sequence(gpu_id=idx)
                        if isinstance(type, list):
                            self.build_sequence_prediction(type=new_type,gpu_id=idx,accK=accK1,nb_class=nb_class1)
                            self.build_single_prediction(gpu_id=idx,accK=accK2,nb_class=nb_class2)
                        else:
                            if type == 'single':
                                self.build_single_prediction(gpu_id=idx,accK=accK2,nb_class=nb_class2)
                            else:
                                self.build_sequence_prediction(type=new_type,gpu_id=idx,accK=accK1,nb_class=nb_class1)
        self.build_model_aggregation()
                             
            
 