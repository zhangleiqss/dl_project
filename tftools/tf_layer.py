import tensorflow as tf
from tensorflow.python.ops import nn_ops
#from tensorflow.python.ops.rnn_cell import RNNCell,LSTMCell,LSTMStateTuple
#from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell,LSTMCell,LSTMStateTuple
from tf_func import *
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from functools import reduce
from tensorflow.python.training import moving_averages
from tensorflow.contrib.rnn.python.ops import rnn_cell
import collections


def _is_sequence(seq):
    return (isinstance(seq, collections.Sequence)
          and not isinstance(seq, six.string_types))

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (_is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not _is_sequence(args):
        args = [args]
  
    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]
  
    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = vs.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable(
          "Bias", [output_size],
          initializer=init_ops.constant_initializer(bias_start))
        return res + bias_term 

#full connected layer
def my_full_connected(input_tensor, output_dim, add_bias=True,
                      layer_name='fully_connected', act=tf.identity, init_std=0.05, 
                      scope=None, reuse=False):
    input_dim = input_tensor.get_shape()[-1].value
    vscope = get_new_variable_scope(layer_name, scope, reuse=reuse)
    with vscope as scope:
        weights = tf.get_variable("W", [input_dim, output_dim], 
                                          initializer=tf.truncated_normal_initializer(stddev=init_std))  
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) 
            if add_bias:
                biases = tf.get_variable("B", [output_dim], initializer=tf.truncated_normal_initializer(stddev=init_std))
                preactivate = preactivate + biases
            activations = act(preactivate, name='activation')
            return activations
    
##embedding_layer
def my_embedding_layer(input_tensor, shape, embedding_size, layer_name='embedding_layer', 
                       #init_mean=0.0, init_std=0.05, 
                       init_scale = 1.0, 
                       scope=None, reuse=False):
    vscope = get_new_variable_scope(layer_name, scope, reuse=reuse)
    with vscope as scope:
        embedding_table = tf.get_variable('embedding_table',[shape, embedding_size],
                                          initializer = tf.random_uniform_initializer(-init_scale,init_scale))
                                          #initializer = tf.truncated_normal_initializer(mean=init_mean,stddev=init_std))
        input_embedding = tf.nn.embedding_lookup(embedding_table, input_tensor)
        return input_embedding

#embedding_layer with non-trainable zero label 
def my_embedding_layer_with_zero_mask(input_tensor, shape, embedding_size, layer_name='embedding_layer', 
                                      #init_mean=0.0, init_std=0.05, 
                                      init_scale = 1.0,
                                      scope=None, reuse=False):
    vscope = get_new_variable_scope(layer_name, scope, reuse=reuse)
    with vscope as scope:
        #mask_embedding_table = tf.get_variable('zero_mask_embedding_table',[1, 100], 
        #                                  initializer = tf.truncated_normal_initializer(mean=init_mean,stddev=init_std), trainable=False)
        mask_embedding_table = tf.get_variable('zero_mask_embedding_table',[1, embedding_size], 
                                          initializer = tf.constant_initializer(0.0), trainable=False)
        embedding_table = tf.get_variable('embedding_table',[shape-1, embedding_size], 
                                          initializer = tf.random_uniform_initializer(-init_scale,init_scale))
                                          #initializer = tf.truncated_normal_initializer(mean=init_mean,stddev=init_std))
        embedding_table = tf.concat([mask_embedding_table, embedding_table],0)
        input_embedding = tf.nn.embedding_lookup(embedding_table, input_tensor)
        return input_embedding

#one-hot layer    
def my_onehot_layer(input_tensor, shape, layer_name='embedding_layer'):
    with tf.name_scope(layer_name):
        input_one_hot = tf.one_hot(input_tensor, shape)
        return input_one_hot   

#
def my_conv_1d(input_tensor, conv_length, n_filters_out, add_bias=True,
               stride_step=1, padding='SAME', layer_name='conv_1d', act=tf.nn.tanh, init_std=0.05,
               bn=False, training=tf.constant(True), scope=None, reuse=False):
    n_filters_in = input_tensor.get_shape()[-1].value
    vscope = get_new_variable_scope(layer_name, scope, reuse=reuse)
    with vscope as scope:
        conv_W = he_uniform('W', [conv_length, n_filters_in, n_filters_out])
        with tf.name_scope('conv_1d_operation'):
            conv_x = tf.nn.conv1d(input_tensor, conv_W, stride=stride_step, padding=padding)
            if add_bias:
                biases = tf.get_variable("B", [n_filters_out], initializer=tf.truncated_normal_initializer(stddev=init_std))
                conv_x = tf.nn.bias_add(conv_x, biases)
            if bn:
                conv_x = my_batch_norm(conv_x, training)
            conv_x = act(conv_x, name='activation')
        return conv_x


def my_atrous_conv_1d(input_tensor, conv_length, n_filters_out, rate, add_bias=True,
                      padding='SAME', layer_name='atrous_conv_1d', act=tf.nn.tanh, 
                      bn=False, training=tf.constant(True), scope=None, reuse=False):
    n_filters_in = input_tensor.get_shape()[-1].value
    vscope = get_new_variable_scope(layer_name, scope, reuse=reuse)
    with vscope as scope:
        conv_W = he_uniform('W', [1, conv_length, n_filters_in, n_filters_out])
        if padding=='SAME':
            pad_len = (conv_length - 1) * rate
            x = tf.expand_dims(tf.pad(input_tensor, [[0, 0], [pad_len, 0], [0, 0]]),1) 
        with tf.name_scope('atrous_conv_1d_operation'):
            conv_x = tf.nn.atrous_conv2d(x, conv_W, rate=rate, padding='VALID')
            if add_bias:
                biases = tf.get_variable("B", [n_filters_out], initializer=tf.truncated_normal_initializer(stddev=init_std))
                conv_x = tf.nn.bias_add(conv_x, biases)
            if bn:
                conv_x = my_batch_norm(conv_x, training)
            conv_x = tf.squeeze(act(conv_x, name='activation'),[1])
        return conv_x


def my_conv_2d(input_tensor, shape, strides=[1,1,1,1], add_bias=True,
               padding='SAME', layer_name='conv_2d',act=tf.nn.tanh,
               bn=False, training=tf.constant(True), scope=None, reuse=False):
    #shape: [kernel_width, kernel_height, n_filters_in, n_filters_out]
    vscope = get_new_variable_scope(layer_name, scope, reuse=reuse)
    with vscope as scope:
        conv_W = he_uniform('W', shape)
        with tf.name_scope('conv_2d_operation'):
            conv_x = tf.nn.conv2d(input_tensor, conv_W, strides=strides, padding=padding)
            if add_bias:
                biases = tf.get_variable("B", [shape[-1]], initializer=tf.truncated_normal_initializer(stddev=init_std))
                conv_x = tf.nn.bias_add(conv_x, biases)
            if bn:
                conv_x = my_batch_norm(conv_x, training)
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

"""
 Highway Network (cf. http://arxiv.org/abs/1505.00387).
  t = sigmoid(Wy + b)
  z = t * g(Wy + b) + (1 - t) * y
  where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
 """
def highway(input_, layer_size=1, bias=0.0, f=tf.nn.relu, layer_name='highway',scope=None, reuse=False):
    output = input_
    size = input_.get_shape()[-1]
    vscope = get_new_variable_scope(layer_name, scope, reuse=reuse)
    with vscope as scope:
        for idx in range(layer_size):
            output = f(_linear(output, size, 0, scope='output_lin_%d' % idx))
            transform_gate = tf.sigmoid(_linear(input_, size, 0, scope='transform_lin_%d' % idx) + bias)
            carry_gate = 1. - transform_gate
            output = transform_gate * output + carry_gate * input_
        return output
        
def my_batch_norm(input_tensor, training, recurrent=False, epsilon=1e-3, decay=0.999, layer_name='bn_layer'):
    if recurrent:
        vscope = get_new_variable_scope(layer_name, scope=layer_name)
    else:
        vscope = get_new_variable_scope(layer_name)       
    x_shape = input_tensor.get_shape()
    axis = list(range(len(x_shape) - 1))
    params_shape = x_shape[-1:]
    with vscope as scope:   
        #print scope.name
        scale = tf.get_variable('scale', params_shape, initializer=tf.constant_initializer(1.0))
        offset = tf.get_variable('offset', params_shape, initializer=tf.constant_initializer(0.0))
        pop_mean = tf.get_variable('pop_mean', params_shape, initializer=tf.constant_initializer(0), trainable=False)
        pop_var = tf.get_variable('pop_var', params_shape, initializer=tf.constant_initializer(1.0), trainable=False)
      
        batch_mean, batch_var = tf.nn.moments(input_tensor, axis)    
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        #train_mean = moving_averages.assign_moving_average(pop_mean, batch_mean, decay)
        #train_var = moving_averages.assign_moving_average(pop_var, batch_var, decay)

        def batch_statistics():
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(input_tensor, batch_mean, batch_var, offset, scale, epsilon)
        def population_statistics():
            return tf.nn.batch_normalization(input_tensor,pop_mean, pop_var, offset, scale, epsilon)
        
        return tf.cond(training, batch_statistics, population_statistics)

