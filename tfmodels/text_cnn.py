import sys
sys.path.append('../tftools')
from tf_object import *

#Convolutional Neural Networks for Sentence Classification
class TextCNN(TFModel):
    def __init__(self, config, sess, current_task_name='textcnn'):
        super(TextCNN, self).__init__(config, sess)
        self.nb_words = config.nb_words
        self.maxlen = config.maxlen
        self.embedding_size = config.embedding_size
        self.init_scale = config.init_scale
        self.current_task_name = current_task_name
        
    def build_input(self):
        with tf.name_scope('input'):
            inputX = tf.placeholder(tf.int32, [None, self.maxlen], name="input_sentence")
            inputLabel = tf.placeholder(tf.int32, [None], name='label')
            tf.add_to_collection(tf.GraphKeys.INPUTS, inputX)
            tf.add_to_collection(tf.GraphKeys.INPUTS, inputLabel)
            self.split_inputX = tf.split(inputX, self.gpu_num, 0)
            self.split_inputLabel = tf.split(inputLabel, self.gpu_num, 0)
        self.__build_global_setting__()
        with tf.name_scope('states_array'):
            self.pool_flat_drop_list = [[] for i in range(0,self.gpu_num)]
        
    def build_cnn_model(self, gpu_id=0, max_conv_len=3, num_filters=128, dropout_keep_prob=0.5):
        with get_new_variable_scope('embedding') as embedding_scope:
            input_embedding = my_embedding_layer_with_zero_mask(self.split_inputX[gpu_id], self.nb_words, self.embedding_size, 
                                  layer_name='embedding_layer', init_scale=self.init_scale)
        pooled_outputs = []
        for conv_length in range(1, max_conv_len+1):
            with get_new_variable_scope('conv') as conv_scope:
                conv = my_conv_1d(input_embedding, conv_length, num_filters, add_bias=True, bn=False, 
                                  padding='VALID', act=tf.nn.relu)
                #conv_list.append(conv)
                pool = tf.nn.max_pool(tf.expand_dims(conv, -1),
                                      ksize=[1, self.maxlen - conv_length + 1, 1, 1],
                                      strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pool)
                
        num_filters_total = num_filters * len(pooled_outputs)
        concat_pool = tf.concat(pooled_outputs,3)
        concat_pool_flat = tf.reshape(concat_pool, [-1, num_filters_total])
        # Add dropout
        with tf.name_scope("dropout"):
            self.pool_flat_drop_list[gpu_id] = tf.nn.dropout(concat_pool_flat, dropout_keep_prob)
    
    def build_prediction(self, gpu_id=0, num_classes=2, accK=1):
        prediction = my_full_connected(self.pool_flat_drop_list[gpu_id], num_classes)
        self.tower_prediction_results.append(tf.nn.softmax(prediction))
        self.params = tf.trainable_variables()[1:]
        with tf.name_scope('loss'): 
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.split_inputLabel[gpu_id], logits=prediction)
            grads, capped_gvs = my_compute_grad(self.opt, loss, self.params, 
                                                clip_type = 'clip_norm', 
                                                max_clip_grad=self.clip_gradients)           
        with tf.name_scope('accuracy'):
            accuracy = tf.to_float(tf.nn.in_top_k(prediction, self.split_inputLabel[gpu_id],k=accK))
        self.__add_to_tower_list__(grads,capped_gvs,loss,accuracy)

                
    def build_model(self, num_classes=2, *args, **kwargs):
        self.build_input()
        for idx, gpu_id in enumerate(self.gpus):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('Tower_%d' % (gpu_id)) as tower_scope:
                    gpu_scope = tf.variable_scope('gpu', reuse=(idx!=0))
                    with gpu_scope as gpu_scope:
                        self.build_cnn_model(gpu_id=idx, *args, **kwargs)
                        self.build_prediction(gpu_id=idx, num_classes=num_classes)                       
        self.build_model_aggregation()                
        

#Character-level Convolutional Networks for Text Classification
class TextMultiCNN(TextCNN):
    def __init__(self, config, sess, current_task_name='text_multi_cnn'):
        super(TextMultiCNN, self).__init__(config, sess, current_task_name)
        
    def build_cnn_model(self, gpu_id=0, num_filters_per_size=64, filter_sizes=[7, 7, 3, 3, 3, 3], 
                        pool_sizes=[3,3,None,None,None,3], fully_layers = [128, 128], dropout_keep_prob=0.5):
        with get_new_variable_scope('embedding') as embedding_scope:
            input_embedding = my_embedding_layer(self.split_inputX[gpu_id], self.nb_words, self.embedding_size, 
                                  layer_name='embedding_layer', init_scale=self.init_scale)
        last_conv = input_embedding
        for idx, conv_length in enumerate(filter_sizes):
            with get_new_variable_scope('conv') as conv_scope:
                conv = my_conv_1d(last_conv, conv_length, num_filters_per_size, add_bias=True, bn=False, 
                                  padding='VALID', act=tf.nn.relu)                
            if pool_sizes[idx] is not None:
                with tf.name_scope("MaxPoolingLayer"):
                    pool = tf.nn.max_pool(tf.expand_dims(conv, -1),
                                          ksize=[1, pool_sizes[idx], 1, 1], strides=[1, pool_sizes[idx], 1, 1],
                                          padding='VALID', name="pool")
                    last_conv = tf.squeeze(pool,[3])
            else:
                last_conv = conv
        last_flatten = my_flatten(last_conv)
        for idx, fc in enumerate(fully_layers):
            flatten = my_full_connected(last_flatten, fc, act=tf.nn.relu)
            last_flatten = tf.nn.dropout(highway(flatten), dropout_keep_prob)
        self.pool_flat_drop_list[gpu_id] = last_flatten
                    

class TextMultiClassCNN(TextMultiCNN):
    def __init__(self, config, sess, current_task_name='text_multi_class_cnn_emotion'):
        super(TextMultiClassCNN, self).__init__(config, sess, current_task_name) 
    
    def build_input(self, num_classes):
        with tf.name_scope('input'):
            inputX = tf.placeholder(tf.int32, [None, self.maxlen], name="input_sentence")
            inputLabel = tf.placeholder(tf.float32, [None, num_classes], name='label')
            tf.add_to_collection(tf.GraphKeys.INPUTS,  inputX)
            tf.add_to_collection(tf.GraphKeys.INPUTS,  inputLabel)
            self.split_inputX = tf.split(inputX, self.gpu_num, 0)
            self.split_inputLabel = tf.split(inputLabel, self.gpu_num, 0)
        self.__build_global_setting__()
        with tf.name_scope('states_array'):
            self.pool_flat_drop_list = [[] for i in range(0,self.gpu_num)]
            
    def build_prediction(self, gpu_id=0, num_classes=6):
        prediction = my_full_connected(self.pool_flat_drop_list[gpu_id], num_classes)
        self.tower_prediction_results.append(tf.nn.softmax(prediction))
        self.params = tf.trainable_variables()[1:]
        with tf.name_scope('loss'): 
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.split_inputLabel[gpu_id], logits=prediction)
            #loss = tf.square(tf.nn.softmax(prediction) - self.split_inputLabel[gpu_id])
            grads, capped_gvs = my_compute_grad(self.opt, loss, self.params, 
                                                clip_type = 'clip_norm', 
                                                max_clip_grad=self.clip_gradients)           
        #this metrics need to be redefined....
        with tf.name_scope('accuracy'):
            accuracy = tf.to_float(tf.nn.in_top_k(prediction, tf.argmax(self.split_inputLabel[gpu_id],axis=1),k=2))
        self.__add_to_tower_list__(grads,capped_gvs,loss,accuracy)
        
    #def build_model(self, num_classes=6, *args, **kwargs):
    #    self.build_input(num_classes)
    #    for idx, gpu_id in enumerate(self.gpus):
    #        with tf.device('/gpu:%d' % gpu_id):
    #            with tf.name_scope('Tower_%d' % (gpu_id)) as tower_scope:
    #                gpu_scope = tf.variable_scope('gpu', reuse=(idx!=0))
    #               with gpu_scope as gpu_scope:
    #                    self.build_cnn_model(gpu_id=idx, *args, **kwargs)
    #                    self.build_prediction(gpu_id=idx, num_classes=num_classes)                       
    #    self.build_model_aggregation()            
                        