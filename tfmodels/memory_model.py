import sys
sys.path.append('../tftools')

from tf_object import *
import tensorflow as tf

#random_uniform or truncated_normal? it's a question.. sigh..
class MemN2NModel(TFModel):
    def __init__(self, config, sess, current_task_name='memn2n_model'): 
        super(MemN2NModel, self).__init__(config, sess)
        self.nb_words = config.nb_words 
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.init_scale = config.init_std
        self.mem_size = config.mem_size #the mem_size (max length) of input sequence
        self.embedding_size = config.embedding_size #embedding size
        self.nhop = config.nhop 
        self.lindim = int(math.floor(config.linear_ratio * self.embedding_size))
        self.current_task_name = current_task_name
        self.tower_attetions = [[] for i in range(0,self.gpu_num)]
        
    def build_input(self):
        with tf.name_scope('input'):
            input_context = tf.placeholder(tf.int32, [None, self.mem_size], name="context")
            output_target = tf.placeholder(tf.int32, [None], name="target")
            self.__add_to_graph_input__([input_context,output_target])
            #tf.add_to_collection(tf.GraphKeys.INPUTS,  input_context)
            #tf.add_to_collection(tf.GraphKeys.INPUTS,  output_target) 
            self.split_input_context = tf.split(input_context, self.gpu_num, 0)
            self.split_output_target = tf.split(output_target, self.gpu_num, 0)
        self.__build_global_setting__()
        with tf.name_scope('state_array'):
            self.hid_list = [[] for i in range(0,self.gpu_num)]
                   
 
    def build_memory(self, gpu_id=0):
        with get_new_variable_scope('memory') as memory_scope:
            Ain_c = my_embedding_layer(self.split_input_context[gpu_id], self.nb_words, self.embedding_size, layer_name='Ain_c', init_scale=self.init_scale)
            with get_new_variable_scope('Ain_t') as scope:
                Ain_t = tf.get_variable('W', [self.mem_size, self.embedding_size], 
                                        initializer = tf.random_uniform_initializer(-self.init_scale,self.init_scale))
                                        #initializer=tf.truncated_normal_initializer(0.0, self.init_std))
            Ain = tf.add(Ain_c, Ain_t) 
            input_q = tf.ones_like(Ain[:,0])/10             
            Cin_c = my_embedding_layer(self.split_input_context[gpu_id], self.nb_words, self.embedding_size, layer_name='Cin_c',init_scale=self.init_scale)
            with get_new_variable_scope('Cin_t') as scope:
                Cin_t = tf.get_variable('W', [self.mem_size, self.embedding_size], 
                                        initializer = tf.random_uniform_initializer(-self.init_scale,self.init_scale))
                                        #initializer=tf.truncated_normal_initializer(0.0, self.init_std))
            Cin = tf.add(Cin_c, Cin_t)    
        with tf.name_scope('hidden_state'):
            #self.hid = []
            #self.hid.append(input_q)
            self.hid_list[gpu_id].append(input_q)
        with get_new_variable_scope('momory_hops') as hops_scope:
            with get_new_variable_scope('hops_h') as scope:
                H = tf.get_variable('W',[self.embedding_size, self.embedding_size], 
                                    initializer = tf.random_normal_initializer(0.0,self.init_std)) 
            for h in xrange(self.nhop):
                hid3dim = tf.reshape(self.hid_list[gpu_id][-1], [-1, 1, self.embedding_size])
                Aout = tf.matmul(hid3dim, Ain, adjoint_b=True)
                Aout_norm = tf.nn.softmax(Aout)
                #add the attention_distribution
                self.tower_attetions[gpu_id] = Aout_norm  
                Cout = tf.matmul(Aout_norm, Cin)
                Cout2dim = tf.reshape(Cout, [-1, self.embedding_size])
                Dout = tf.add(tf.matmul(self.hid_list[gpu_id][-1], H), Cout2dim)
                #linear relu for a part of hidden unit
                if self.lindim == self.embedding_size:
                    self.hid_list[gpu_id].append(Dout)
                elif self.lindim == 0:
                    self.hid_list[gpu_id].append(tf.nn.relu(Dout))
                else:
                    F = tf.slice(Dout, [0, 0], [-1, self.lindim])
                    G = tf.slice(Dout, [0, self.lindim], [-1, self.embedding_size-self.lindim])
                    K = tf.nn.relu(G)
                    self.hid_list[gpu_id].append(tf.concat([F, K],1)) 
            
            
    def build_prediction(self, type='self', accK=5, nb_class=None, gpu_id=0):
        if type == 'self':
            nb_class = self.nb_words
        elif nb_class == None:
            raise Exception("nb_class must be given")
        with get_new_variable_scope('prediction') as pred_scope:
            prediction = my_full_connected(self.hid_list[gpu_id][-1], nb_class, 
                                           add_bias=False, layer_name='fc', 
                                           act=tf.identity, init_std=self.init_std)
            self.tower_prediction_results.append(tf.nn.softmax(prediction))
            self.params = tf.trainable_variables()[1:]
        with tf.name_scope('train'):
            ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.split_output_target[gpu_id], logits=prediction)
            grads, capped_gvs = my_compute_grad(self.opt, ce_loss, self.params, 
                                                clip_type = 'clip_norm', 
                                                max_clip_grad=self.clip_gradients)            
        with tf.name_scope('accuracy'):
                accuracy = tf.to_float(tf.nn.in_top_k(prediction, self.split_output_target[gpu_id],k=accK))   
        self.__add_to_tower_list__(grads, capped_gvs, ce_loss, accuracy)
            
    def build_model(self, type='self', accK=5, nb_class=None, export_attention=False):
        self.build_input()
        for idx, gpu_id in enumerate(self.gpus):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('Tower_%d' % (gpu_id)) as tower_scope:
                    gpu_scope = tf.variable_scope('gpu', reuse=(idx!=0))
                    with gpu_scope as gpu_scope:
                        self.build_memory(gpu_id=idx)
                        self.build_prediction(type=type,accK=accK,nb_class=nb_class,gpu_id=idx)
                     
        self.build_model_aggregation()
        if export_attention:
            self.prediction_results = [self.prediction_results, tf.concat(self.tower_attetions,0)]
        
        
class MemN2NUserModel(MemN2NModel):
    def __init__(self, config, sess, current_task_name='memn2n_user_model'): 
        super(MemN2NUserModel, self).__init__(config, sess, current_task_name)
        self.q_size = 2000
    
    def build_input(self):
        with tf.name_scope('input'):
            input_context = tf.placeholder(tf.int32, [None, self.mem_size], name="context")
            input_viewthru = tf.placeholder(tf.int32, [None, self.mem_size], name="viewthru")
            #input_hourseq = tf.placeholder(tf.int32, [None, self.mem_size], name="hourseq")
            input_question = tf.placeholder(tf.int32, [None], name="question")
            output_target = tf.placeholder(tf.int32, [None], name="target")
            self.__add_to_graph_input__([input_context,input_viewthru,input_question,output_target])
            self.split_input_context = tf.split(input_context, self.gpu_num, 0)
            self.split_input_viewthru = tf.split(input_viewthru, self.gpu_num, 0)
            #self.split_input_hourseq = tf.split(input_hourseq, self.gpu_num, 0)
            self.split_input_question = tf.split(input_question, self.gpu_num, 0)
            self.split_output_target = tf.split(output_target, self.gpu_num, 0)
        with tf.name_scope('global'):
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            self.lr = tf.Variable(self.learning_rate)
            self.opt = tf.train.GradientDescentOptimizer(self.lr)
        with tf.name_scope('state_array'):
            self.hid_list = [[] for i in range(0,self.gpu_num)]
    
    def build_memory(self, gpu_id=0):
        with get_new_variable_scope('memory') as memory_scope:
            Ain_c = my_embedding_layer_with_zero_mask(self.split_input_context[gpu_id], self.nb_words, self.embedding_size, layer_name='Ain_c', init_scale=self.init_scale)
            Ain_c_vt = my_embedding_layer_with_zero_mask(self.split_input_viewthru[gpu_id], 101, self.embedding_size, layer_name='Ain_c_vt', init_scale=self.init_scale)
            Ain_c = tf.add(Ain_c, Ain_c_vt)
            #Ain_c_h = my_embedding_layer_with_zero_mask(self.split_input_hourseq[gpu_id], 1441, self.embedding_size, layer_name='Ain_c_h', init_std=self.init_std)
            #Ain_c = tf.add(Ain_c, Ain_c_h)            
            with get_new_variable_scope('Ain_t') as scope:
                Ain_t = tf.get_variable('W', [self.mem_size, self.embedding_size],
                                        initializer = tf.random_uniform_initializer(-self.init_scale,self.init_scale))
                                        #initializer=tf.truncated_normal_initializer(0.0, self.init_std))
            Ain = tf.add(Ain_c, Ain_t)
            input_q = my_embedding_layer(self.split_input_question[gpu_id], self.q_size, self.embedding_size, layer_name='questions', init_scale=self.init_scale)
            #input_q = tf.ones_like(Ain[:,0])/10             
            Cin_c = my_embedding_layer_with_zero_mask(self.split_input_context[gpu_id], self.nb_words, self.embedding_size, layer_name='Cin_c',init_scale=self.init_scale)
            Cin_c_vt = my_embedding_layer_with_zero_mask(self.split_input_viewthru[gpu_id], 101, self.embedding_size, layer_name='Cin_c_vt', init_scale=self.init_scale)
            Cin_c = tf.add(Cin_c, Cin_c_vt)
            #Cin_c_h = my_embedding_layer_with_zero_mask(self.split_input_hourseq[gpu_id], 1441, self.embedding_size, layer_name='Cin_c_h', init_std=self.init_std)
            #Cin_c = tf.add(Cin_c, Cin_c_h)
         
            with get_new_variable_scope('Cin_t') as scope:
                Cin_t = tf.get_variable('W', [self.mem_size, self.embedding_size], 
                                        initializer = tf.random_uniform_initializer(-self.init_scale,self.init_scale))
                                        #initializer=tf.truncated_normal_initializer(0.0, self.init_std))
            Cin = tf.add(Cin_c, Cin_t)    
        with tf.name_scope('hidden_state'):
            self.hid_list[gpu_id].append(input_q)
        with get_new_variable_scope('momory_hops') as hops_scope:
            with get_new_variable_scope('hops_h') as scope:
                H = tf.get_variable('W',[self.embedding_size, self.embedding_size], 
                                    initializer = tf.random_normal_initializer(0.0,self.init_std)) 
            for h in xrange(self.nhop):
                hid3dim = tf.reshape(self.hid_list[gpu_id][-1], [-1, 1, self.embedding_size])
                Aout = tf.matmul(hid3dim, Ain, adjoint_b=True)
                Aout_norm = tf.nn.softmax(Aout)
                Cout = tf.matmul(Aout_norm, Cin)
                Cout2dim = tf.reshape(Cout, [-1, self.embedding_size])
                Dout = tf.add(tf.matmul(self.hid_list[gpu_id][-1], H), Cout2dim)
                #linear relu for a part of hidden unit
                if self.lindim == self.embedding_size:
                    self.hid_list[gpu_id].append(Dout)
                elif self.lindim == 0:
                    self.hid_list[gpu_id].append(tf.nn.relu(Dout))
                else:
                    F = tf.slice(Dout, [0, 0], [-1, self.lindim])
                    G = tf.slice(Dout, [0, self.lindim], [-1, self.embedding_size-self.lindim])
                    K = tf.nn.relu(G)
                    self.hid_list[gpu_id].append(tf.concat([F, K],1)) 
            
            
    def run(self, input_list, train_idxs, test_idxs, run_type='self', shuffle=True):
        best_train_metric = self.__worst_metric__()
        best_test_metric = self.__worst_metric__()
        input_var = tf.get_collection(tf.GraphKeys.INPUTS)
        best_auc = 0.0
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
            epochTestLoss, epochTestMetric, result = self.model_run(input_list, test_idxs, run_type,
                                                               mode='test', shuffle=False, save_metric=True)
            
            result = np.concatenate(result)
            y = np.array(input_list[-1][test_idxs])[0:len(result)]
            pred = result[:,1]
            auc = roc_auc_score(y, pred)
            if auc > best_auc:
                best_auc = auc
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
                'valid_perplexity': math.exp(epochTestLoss),
                'best_auc': best_auc
                }
            print(state)
            
            if self.lr_annealing:
                if len(self.log_loss) > 1 and self.log_loss[epoch][1] > self.log_loss[epoch-1][1] * 0.9999:
                    self.learning_rate = self.learning_rate / self.lr_annealing_value
                    self.lr.assign(self.learning_rate).eval()
            if self.lr_annealing and self.learning_rate < self.lr_stop_value: break

