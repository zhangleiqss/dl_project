import sys
sys.path.append('../tfmodels')
from sequential_model import *

class User2VecModel(SequentialModel):
    def __init__(self, config, sess, current_task_name='user2vec_model'):
        super(User2VecModel, self).__init__(config, sess, current_task_name)
        self.one_hot_embedding = config.one_hot_embedding
        
        
    def build_input(self):
        super(User2VecModel, self).build_input()
        with tf.name_scope('input'):
            self.input_is_movie = tf.placeholder(tf.float32, [None, self.maxlen], name='is_movie')
            self.input_time = tf.placeholder(tf.float32, [None, self.maxlen], name='time')
            self.__add_to_graph_input__([self.input_is_movie, self.input_time])
            self.split_input_is_movie = tf.split(tf.expand_dims(self.input_is_movie,2), self.gpu_num, 0)
            self.split_input_time = tf.split(tf.expand_dims(self.input_time,2), self.gpu_num, 0)
    
    def __build_embedding_layer__(self, gpu_id=0):
        with get_new_variable_scope('embedding') as embedding_scope:
            if self.one_hot_embedding:
                input_show_embedding = my_onehot_layer(self.split_inputX[gpu_id], self.nb_words)
            else:
                input_show_embedding = my_embedding_layer(self.split_inputX[gpu_id], self.nb_words, 
                                                          self.embedding_size, layer_name='embedding_layer', 
                                                          init_scale=self.init_scale)
            self.input_embedding = tf.concat([input_show_embedding, self.split_input_is_movie[gpu_id], 
                                              self.split_input_time[gpu_id]],2)
                
    def build_model(self, type='self', accK=5, nb_class=None):
        super(User2VecModel, self).build_model(type, accK, nb_class)
        