import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.rnn_cell import RNNCell,LSTMCell,LSTMStateTuple
from tf_func import *

#my full connected layer
def my_full_connected(input_tensor, input_dim, output_dim, add_bias=True,
                      reuse=None, layer_name='fully_connected', act=tf.identity, init_std=0.05):
    
    with tf.name_scope(layer_name):
        if reuse==None:
            with tf.variable_scope('weights'):
                #weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=init_std))
                weights = tf.Variable(tf.random_normal([input_dim, output_dim], stddev=init_std)) 
        else:
            with tf.variable_scope(layer_name + '/weights', reuse=reuse):
                weights = tf.get_variable("W", [input_dim, output_dim], 
                                          initializer=tf.truncated_normal_initializer(stddev=init_std))  
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) 
            if add_bias:
                if reuse==None: 
                    with tf.variable_scope('biases'):
                        #biases =tf.Variable(tf.truncated_normal([output_dim], stddev=init_std))
                        biases =tf.Variable(tf.random_normal([output_dim], stddev=init_std))
                else:
                    with tf.variable_scope(layer_name + '/biases', reuse=reuse):
                        biases = tf.get_variable("B", [output_dim], initializer=tf.truncated_normal_initializer())
                preactivate = preactivate + biases
        activations = act(preactivate, name='activation')
        return activations
    
def my_embedding_layer(input_tensor, shape, embedding_size, 
                       layer_name='embedding_layer', init_mean=0.0, init_std=0.05):
    
    with tf.name_scope(layer_name):
        with tf.name_scope('embedding_table'):
            embedding_table = tf.Variable(tf.truncated_normal([shape, embedding_size], init_mean, init_std))
        input_embedding = tf.nn.embedding_lookup(embedding_table, input_tensor)
        return input_embedding

def my_onehot_layer(input_tensor, shape, layer_name='embedding_layer'):
    with tf.name_scope(layer_name):
        input_one_hot = tf.one_hot(input_tensor, shape)
        return input_one_hot   

def my_conv_1d(input_tensor, conv_length, n_filters_out, add_bias=True,
               stride_step=1, padding='SAME', layer_name='conv_1d', act=tf.nn.tanh):
    n_filters_in = input_tensor.get_shape()[-1].value
    with tf.name_scope(layer_name):
        with tf.name_scope('filter_weights'):
            #conv_W = tf.Variable(tf.random_normal([conv_length, n_filters_in, n_filters_out], stddev=init_std))
            conv_W = he_uniform('W', [conv_length, n_filters_in, n_filters_out])
        with tf.name_scope('conv_1d_operation'):
            conv_x = tf.nn.conv1d(input_tensor, conv_W, stride=stride_step, padding=padding)
            if add_bias:
                with tf.name_scope('filter_biases'):
                    conv_b = tf.Variable(tf.random_normal([n_filters_out], stddev=init_std))
                    conv_x = tf.nn.bias_add(conv_x, conv_b)
        conv_x = act(conv_x, name='activation')
        return conv_x

def my_atrous_conv_1d(input_tensor, conv_length, n_filters_out, rate, add_bias=True,
               padding='SAME', layer_name='atrous_conv_1d', act=tf.nn.tanh):
    n_filters_in = input_tensor.get_shape()[-1].value
    with tf.name_scope(layer_name):
        with tf.name_scope('filter_weights'):
            #conv_W = tf.Variable(tf.random_normal([conv_length, n_filters_in, n_filters_out], stddev=init_std))
            conv_W = he_uniform('W', [1, conv_length, n_filters_in, n_filters_out])
        if padding=='SAME':
            pad_len = (conv_length - 1) * rate
            x = tf.expand_dims(tf.pad(input_tensor, [[0, 0], [pad_len, 0], [0, 0]]),1)    
        with tf.name_scope('atrous_conv_1d_operation'):
            conv_x = tf.nn.atrous_conv2d(x, conv_W, rate=rate, padding='VALID')
            if add_bias:
                with tf.name_scope('filter_biases'):
                    conv_b = tf.Variable(tf.random_normal([n_filters_out], stddev=init_std))
                    conv_x = tf.nn.bias_add(conv_x, conv_b)
        conv_x = tf.squeeze(act(conv_x, name='activation'),[1])
        return conv_x


def my_conv_2d(input_tensor, shape, strides=[1,1,1,1], add_bias=True,
               padding='SAME', layer_name='conv_2d',act=tf.nn.tanh, init_std=0.05):
    #shape: [kernel_width, kernel_height, n_filters_in, n_filters_out]
    with tf.name_scope(layer_name):
        with tf.name_scope('filter_weights'):
            conv_W = tf.Variable(tf.random_normal(shape, stddev=init_std))     
        with tf.name_scope('conv_2d_operation'):
            conv_x = tf.nn.conv2d(input_tensor, conv_W, strides=strides, padding=padding)
            if add_bias:
                with tf.name_scope('filter_biases'):
                    conv_b = tf.Variable(tf.random_normal([shape[-1]], stddev=init_std))
                    conv_x = tf.nn.bias_add(conv_x, conv_b)
        conv_x = act(conv_x, name='activation')
        return conv_x


def my_pool_layer_2d(input_tensor, k, stride = None, padding='VALID', layer_name='pool_2d', act=tf.nn.max_pool):
    with tf.name_scope(layer_name):
        if stride is None:
            stride = k
        pool = act(input_tensor, ksize=[1,k,k,1], strides=[1,stride,stride,1], padding=padding)
        return pool

def my_flatten(input_tensor, layer_name='flatten'):
    with tf.name_scope(layer_name):
        shape = len(input_tensor.get_shape())
        if shape > 2:
            flatten_shape = reduce(lambda x, y: x*y, [input_tensor.get_shape()[i].value for i in range(1,shape)])
            flatten_layer = tf.reshape(input_tensor,[-1,flatten_shape])
            return flatten_layer
        else:
            return input_tensor
        
def my_batch_norm(input_tensor, training, epsilon=1e-3, decay=0.99, layer_name='bn_layer'):
    with tf.name_scope(layer_name):
        x_shape = input_tensor.get_shape()
        axis = list(range(len(x_shape) - 1))
        params_shape = x_shape[-1:]
        scale = tf.Variable(tf.ones(params_shape), name='scale')
        beta = tf.Variable(tf.zeros(params_shape), name='beta')
        pop_mean = tf.Variable(tf.zeros(params_shape), trainable=False, name='pop_mean')
        pop_var = tf.Variable(tf.ones(params_shape), trainable=False, name='pop_var')
        batch_mean, batch_var = tf.nn.moments(input_tensor, axis)
        
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        
        def batch_statistics():
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(input_tensor,batch_mean, batch_var, beta, scale, epsilon)
        def population_statistics():
            return tf.nn.batch_normalization(input_tensor,pop_mean, pop_var, beta, scale, epsilon)
        
        return tf.cond(training, batch_statistics, population_statistics)
