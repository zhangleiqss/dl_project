import sys
sys.path.append('../tftools')

from tf_object import *
import tensorflow as tf


class MemN2NModel(TFModel):
    def __init__(self, config, sess, current_task_name='memn2n_model'): 
        super(MemN2NModel, self).__init__(config, sess)
        self.nb_words = config.nb_words 
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.mem_size = config.mem_size #the mem_size (max length) of input sequence
        self.embedding_size = config.embedding_size #embedding size
        self.nhop = config.nhop 
        self.lindim = int(math.floor(config.linear_ratio * self.embedding_size))
        self.current_task_name = current_task_name
        
    def build_memory(self):
        with tf.device(self.gpu_option): 
            with tf.name_scope('input'):
                self.input_context = tf.placeholder(tf.int32, [None, self.mem_size], name="context")
                self.output_target = tf.placeholder(tf.int32, [None], name="target")
                tf.add_to_collection(tf.GraphKeys.INPUTS,  self.input_context)
                tf.add_to_collection(tf.GraphKeys.INPUTS,  self.output_target) 
            with tf.name_scope('memory'):        
                Ain_c = my_embedding_layer(self.input_context, self.nb_words, self.embedding_size, layer_name='Ain_c', init_std=self.init_std)
                Ain_t = tf.Variable(tf.truncated_normal([self.mem_size, self.embedding_size], 0.0, self.init_std), name='Ain_t')
                Ain = tf.add(Ain_c, Ain_t)
            
                input_q = tf.ones_like(Ain[:,0])/10
            
                Cin_c = my_embedding_layer(self.input_context, self.nb_words, self.embedding_size, layer_name='Cin_c',init_std=self.init_std)
                Cin_t = tf.Variable(tf.truncated_normal([self.mem_size, self.embedding_size], 0.0, self.init_std), name='Cin_t')
                Cin = tf.add(Cin_c, Cin_t)
            with tf.name_scope('hidden_state'):
                self.hid = []
                self.hid.append(input_q)
                
            with tf.name_scope('momory_hops'):
                H = tf.Variable(tf.random_normal([self.embedding_size, self.embedding_size], stddev=self.init_std))
                for h in range(self.nhop):
                    hid3dim = tf.reshape(self.hid[-1], [-1, 1, self.embedding_size])
                    Aout = tf.batch_matmul(hid3dim, Ain, adj_y=True)
                    Aout_norm = tf.nn.softmax(Aout)
                    Cout = tf.batch_matmul(Aout_norm, Cin)
                    Cout2dim = tf.reshape(Cout, [-1, self.embedding_size])
                    Dout = tf.add(tf.matmul(self.hid[-1], H), Cout2dim)
                    #linear relu for a part of hidden unit
                    if self.lindim == self.embedding_size:
                        self.hid.append(Dout)
                    elif self.lindim == 0:
                        self.self.hid.append(tf.nn.relu(Dout))
                    else:
                        F = tf.slice(Dout, [0, 0], [-1, self.lindim])
                        G = tf.slice(Dout, [0, self.lindim], [-1, self.embedding_size-self.lindim])
                        K = tf.nn.relu(G)
                        self.hid.append(tf.concat(1, [F, K]))
            
    def build_prediction(self, type='self', accK=5):
        with tf.device(self.gpu_option):
            with tf.name_scope('prediction'):
                self.prediction = my_full_connected(self.hid[-1], self.embedding_size, self.nb_words, 
                                              layer_name='fc', add_bias=False, 
                                              act=tf.identity, init_std=self.init_std)
                params = tf.trainable_variables()
            with tf.name_scope('train'):
                self.lr = tf.Variable(self.learning_rate)
                opt = tf.train.GradientDescentOptimizer(self.lr)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.prediction, self.output_target)
                optim, grads, capped_gvs = my_minimize_loss(opt, loss, params, 
                                                            clip_type = 'clip_norm', 
                                                            max_clip_grad=self.clip_gradients)
            with tf.name_scope('accuracy'):
                accuracy = tf.to_float(tf.nn.in_top_k(self.prediction, self.output_target,k=accK))
            self.__add_details__(loss, params, optim, grads, capped_gvs, accuracy)
            self.fetch_dict['self_prediction'] = [optim, loss, accuracy]
            
