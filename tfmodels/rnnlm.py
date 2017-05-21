#from text_utils import *
import h5py
import sys
sys.path.append('../tftools')
from tf_object import *

from tensorflow.contrib.rnn.python.ops import core_rnn_cell as rnn_cell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn,dynamic_rnn
from sklearn.model_selection import train_test_split
import pickle as pkl


class RNNLMModel(TFModel):
    def __init__(self, config, sess, current_task_name='rnnlm'):
        super(RNNLMModel, self).__init__(config, sess)
        
        self.nb_words = config.nb_words #term number in input sequence
        self.maxlen = config.maxlen #the max length of input sequence
        self.num_layers = config.num_layers #the number of rnn layers
        self.embedding_size = config.embedding_size  #word embedding size
        self.hidden_size = config.hidden_size #rnn hidden size
        self.keep_prob = config.keep_prob #keep probability of drop out
        self.init_std = config.init_std
        self.init_scale = config.init_scale
        
        #for self prediction
        self.input_params = None
        self.current_task_name = current_task_name
        
    def build_input(self):
        with tf.name_scope('input'):
            #input and seqLen
            self.inputX = tf.placeholder(tf.int32, [None, self.maxlen], name="inputX")
            self.seqLengths = tf.placeholder(tf.int64, [None], name="seqLengths")
            self.__add_to_graph_input__([self.inputX, self.seqLengths])
            self.split_inputX = tf.split(self.inputX, self.gpu_num, 0)
            self.split_seqLengths = tf.split(self.seqLengths, self.gpu_num, 0)
        #global setting
        self.__build_global_setting__()
        #state list for each gpu
        with tf.name_scope('states_array'):
            self.state_list = [[] for i in range(0,self.gpu_num)]
            self.output_list = [[] for i in range(0,self.gpu_num)]
    
    
    def build_output(self, type='self'):
        with tf.name_scope('output'):
            if type == 'self':
                self.targets = tf.concat([tf.slice(self.inputX,[0,1],[-1,-1]),tf.zeros_like(self.inputX[:,0:1])], 1)
            else:
                self.targets = tf.placeholder(tf.int32, [None, self.maxlen], name="targets")
                self.__add_to_graph_input__([self.targets])
            self.split_targets = tf.split(self.targets, self.gpu_num, 0)
    
    def __build_embedding_layer__(self, gpu_id=0):
         #embedding layer
        with get_new_variable_scope('embedding') as embedding_scope:
            self.input_embedding = my_embedding_layer(self.split_inputX[gpu_id], self.nb_words, self.embedding_size, 
                                                 layer_name='embedding_layer', init_scale=self.init_scale)
     
    def build_input_sequence(self, gpu_id=0, reuse=None):
        #embedding layer
        self.__build_embedding_layer__()
        with get_new_variable_scope('rnn_lstm') as rnn_scope:
            single_cell = rnn_cell.LSTMCell(self.hidden_size, use_peepholes=True, state_is_tuple=True, reuse=reuse)
            single_cell = rnn_cell.DropoutWrapper(single_cell, input_keep_prob=self.keep_prob, 
                                                  output_keep_prob=self.keep_prob)
            cell = rnn_cell.MultiRNNCell([single_cell] * self.num_layers, state_is_tuple=True)
            self.state_list[gpu_id], self.output_list[gpu_id] = dynamic_rnn(cell, self.input_embedding, 
                                                                self.split_seqLengths[gpu_id], dtype=tf.float32)              
        if self.input_params is None:
            self.input_params = tf.trainable_variables()[1:]
        
    def build_sequence_prediction(self, type='self', gpu_id=0, nb_class=None, accK=5):
        if nb_class is None:
            nb_class = self.nb_words
        with get_new_variable_scope('prediction') as pred_scope:    
            logits = my_conv_1d(self.state_list[gpu_id], 1,  nb_class, add_bias=True, bn=False, act=tf.identity)
            self.tower_prediction_results.append(logits)
            if self.params is None:
                self.params = tf.trainable_variables()[1:]     
        with tf.name_scope('loss'): 
            loss = variable_seq_loss(logits, self.split_targets[gpu_id], lm_flag=False) 
            grads, capped_gvs = my_compute_grad(self.opt, loss, self.params, 
                                                clip_type = 'clip_norm', 
                                                max_clip_grad=self.clip_gradients)
        with tf.name_scope('accuracy'):
            accuracy = varibale_topk_accuracy(logits, self.split_targets[gpu_id], k=accK, lm_flag=False)
            #accuracy = tf.to_float(tf.nn.in_top_k(prediction, self.split_output_target[gpu_id],k=accK))
        self.__add_to_tower_list__(grads,capped_gvs,loss,accuracy,type)

    def build_model(self, type='self', accK=5, nb_class=None):
        self.build_input()
        self.build_output(type)
        for idx, gpu_id in enumerate(self.gpus):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('Tower_%d' % (gpu_id)) as tower_scope:
                    reuse = (idx!=0)
                    gpu_scope = tf.variable_scope('gpu', reuse=reuse)
                    with gpu_scope as gpu_scope:
                        self.build_input_sequence(gpu_id=idx, reuse=reuse)
                        self.build_sequence_prediction(type=type,gpu_id=idx,accK=accK,nb_class=nb_class)
        self.build_model_aggregation()

