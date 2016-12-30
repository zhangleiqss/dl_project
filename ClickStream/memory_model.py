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
    
    
    def build_input(self):
        with tf.name_scope('input'):
            input_context = tf.placeholder(tf.int32, [None, self.mem_size], name="context")
            output_target = tf.placeholder(tf.int32, [None], name="target")
            tf.add_to_collection(tf.GraphKeys.INPUTS,  input_context)
            tf.add_to_collection(tf.GraphKeys.INPUTS,  output_target) 
            self.split_input_context = tf.split(0, self.gpu_num, input_context)
            self.split_output_target = tf.split(0, self.gpu_num, output_target)
        with tf.name_scope('global'):
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            self.lr = tf.Variable(self.learning_rate)
            self.opt = tf.train.GradientDescentOptimizer(self.lr)
    
    def build_memory(self, gpu_id=0):
        with get_new_variable_scope('memory') as memory_scope:
            Ain_c = my_embedding_layer(self.split_input_context[gpu_id], self.nb_words, self.embedding_size, layer_name='Ain_c', init_std=self.init_std)
            with get_new_variable_scope('Ain_t') as scope:
                Ain_t = tf.get_variable('W', [self.mem_size, self.embedding_size], 
                                        initializer=tf.truncated_normal_initializer(0.0, self.init_std))
            Ain = tf.add(Ain_c, Ain_t) 
            input_q = tf.ones_like(Ain[:,0])/10             
            Cin_c = my_embedding_layer(self.split_input_context[gpu_id], self.nb_words, self.embedding_size, layer_name='Cin_c',init_std=self.init_std)
            with get_new_variable_scope('Cin_t') as scope:
                Cin_t = tf.get_variable('W', [self.mem_size, self.embedding_size], 
                                        initializer=tf.truncated_normal_initializer(0.0, self.init_std))
            Cin = tf.add(Cin_c, Cin_t)    
        with tf.name_scope('hidden_state'):
            self.hid = []
            self.hid.append(input_q)
        with get_new_variable_scope('momory_hops') as hops_scope:
            with get_new_variable_scope('hops_h') as scope:
                H = tf.get_variable('W',[self.embedding_size, self.embedding_size], 
                                    initializer = tf.random_normal_initializer(0.0,self.init_std)) 
            for h in xrange(self.nhop):
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
            
            
    def build_prediction(self, type='self', accK=5, nb_class=None, gpu_id=0):
        if type == 'self':
            nb_class = self.nb_words
        elif nb_class == None:
            raise Exception("nb_class must be given")
        with get_new_variable_scope('prediction') as pred_scope:
            self.prediction = my_full_connected(self.hid[-1], self.embedding_size, nb_class, 
                                           layer_name='fc', add_bias=False, 
                                           act=tf.identity, init_std=self.init_std)
            self.params = tf.trainable_variables()[1:]
        with tf.name_scope('train'):
            ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.prediction, self.split_output_target[gpu_id])
            grads, capped_gvs = my_compute_grad(self.opt, ce_loss, self.params, 
                                                clip_type = 'clip_norm', 
                                                max_clip_grad=self.clip_gradients)
            
            self.tower_grads.append(grads)
            self.tower_capped_gvs.append(capped_gvs)
            self.tower_loss.append(ce_loss)
        with tf.name_scope('accuracy'):
            accuracy = tf.to_float(tf.nn.in_top_k(self.prediction, self.split_output_target[gpu_id],k=accK))
            self.tower_metrics.append(accuracy)
            
    def build_model(self, type='self', accK=5, nb_class=None):
        self.build_input()
        for idx, gpu_id in enumerate(self.gpus):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('Tower_%d' % (gpu_id)) as tower_scope:
                    gpu_scope = tf.variable_scope('gpu', reuse=(idx!=0))
                    with gpu_scope as gpu_scope:
                        self.build_memory(gpu_id=gpu_id)
                        self.build_prediction(type=type,accK=accK,nb_class=nb_class,gpu_id=gpu_id)
                     
        grads_avg = average_gradients(self.tower_grads)
        capped_gvs_avg = average_gradients(self.tower_capped_gvs)
        loss = tf.concat(0, self.tower_loss)
        accuracy = tf.concat(0, self.tower_metrics)
        train_op = self.opt.apply_gradients(capped_gvs_avg, global_step=self.global_step) 
        self.__add_details__(loss, self.params, train_op, grads_avg, capped_gvs_avg, accuracy)  
        self.fetch_dict['self_prediction'] = [train_op, loss, accuracy]
            
