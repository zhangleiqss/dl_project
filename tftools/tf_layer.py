import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.rnn_cell import RNNCell,LSTMCell,LSTMStateTuple
from tf_func import *
from utils import *
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops

#my full connected layer
def my_full_connected(input_tensor, input_dim, output_dim, add_bias=True,
                      layer_name='fully_connected', act=tf.identity, init_std=0.05, scope=None, reuse=False):
    vscope = get_new_variable_scope(scope, layer_name, reuse=reuse)
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
    

def my_embedding_layer(input_tensor, shape, embedding_size, layer_name='embedding_layer', 
                       init_mean=0.0, init_std=0.05, scope = None, reuse=False):
    vscope = get_new_variable_scope(scope, layer_name, reuse=reuse)
    with vscope as scope:
        embedding_table = tf.get_variable('embedding_table',[shape, embedding_size], 
                                          initializer = tf.truncated_normal_initializer(mean=init_mean,stddev=init_std))
        input_embedding = tf.nn.embedding_lookup(embedding_table, input_tensor)
        return input_embedding


def my_onehot_layer(input_tensor, shape, layer_name='embedding_layer'):
    with tf.name_scope(layer_name):
        input_one_hot = tf.one_hot(input_tensor, shape)
        return input_one_hot   

def my_conv_1d(input_tensor, conv_length, n_filters_out, add_bias=True,
               stride_step=1, padding='SAME', layer_name='conv_1d', act=tf.nn.tanh, scope = None, reuse=False):
    n_filters_in = input_tensor.get_shape()[-1].value
    vscope = get_new_variable_scope(scope, layer_name, reuse=reuse)
    with vscope as scope:
        conv_W = he_uniform('W', [conv_length, n_filters_in, n_filters_out])
        with tf.name_scope('conv_1d_operation'):
            conv_x = tf.nn.conv1d(input_tensor, conv_W, stride=stride_step, padding=padding)
            if add_bias:
                biases = tf.get_variable("B", [n_filters_out], initializer=tf.truncated_normal_initializer(stddev=init_std))
                conv_x = tf.nn.bias_add(conv_x, conv_b)
            conv_x = act(conv_x, name='activation')
        return conv_x


def my_atrous_conv_1d(input_tensor, conv_length, n_filters_out, rate, add_bias=True,
               padding='SAME', layer_name='atrous_conv_1d', act=tf.nn.tanh, scope=None, reuse=False):
    n_filters_in = input_tensor.get_shape()[-1].value
    vscope = get_new_variable_scope(scope, layer_name, reuse=reuse)
    with vscope as scope:
        conv_W = he_uniform('W', [1, conv_length, n_filters_in, n_filters_out])
        if padding=='SAME':
            pad_len = (conv_length - 1) * rate
            x = tf.expand_dims(tf.pad(input_tensor, [[0, 0], [pad_len, 0], [0, 0]]),1) 
        with tf.name_scope('atrous_conv_1d_operation'):
            conv_x = tf.nn.atrous_conv2d(x, conv_W, rate=rate, padding='VALID')
            if add_bias:
                biases = tf.get_variable("B", [n_filters_out], initializer=tf.truncated_normal_initializer(stddev=init_std))
                conv_x = tf.nn.bias_add(conv_x, conv_b)
            conv_x = tf.squeeze(act(conv_x, name='activation'),[1])
        return conv_x


def my_conv_2d(input_tensor, shape, strides=[1,1,1,1], add_bias=True,
               padding='SAME', layer_name='conv_2d',act=tf.nn.tanh):
    #shape: [kernel_width, kernel_height, n_filters_in, n_filters_out]
    vscope = get_new_variable_scope(scope, layer_name, reuse=reuse)
    with vscope as scope:
        conv_W = he_uniform('W', shape)
        with tf.name_scope('conv_2d_operation'):
            conv_x = tf.nn.conv2d(input_tensor, conv_W, strides=strides, padding=padding)
            if add_bias:
                biases = tf.get_variable("B", [shape[-1]], initializer=tf.truncated_normal_initializer(stddev=init_std))
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
        
def my_batch_norm(input_tensor, training, recurrent=False, epsilon=1e-3, decay=0.999, layer_name='bn_layer'):
    if recurrent:
        vscope = get_new_variable_scope(None,scope=layer_name)
    else:
        vscope = get_new_variable_scope(layer_name)
    x_shape = input_tensor.get_shape()
    axis = list(range(len(x_shape) - 1))
    params_shape = x_shape[-1:]
    with vscope as scope:    
        scale = tf.get_variable('scale', params_shape, initializer=tf.constant_initializer(1.0))
        offset = tf.get_variable('offset', params_shape, initializer=tf.constant_initializer(0.0))
        pop_mean = tf.get_variable('pop_mean', params_shape, initializer=tf.constant_initializer(0), trainable=False)
        pop_var = tf.get_variable('pop_var', params_shape, initializer=tf.constant_initializer(1.0), trainable=False)
      
        batch_mean, batch_var = tf.nn.moments(input_tensor, axis)    
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        
        def batch_statistics():
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(input_tensor,batch_mean, batch_var, offset, scale, epsilon)
        def population_statistics():
            return tf.nn.batch_normalization(input_tensor,pop_mean, pop_var, offset, scale, epsilon)
        
        return tf.cond(training, batch_statistics, population_statistics)

###########################################################################################################################    
class BNLSTMCell(LSTMCell):
    def __init__(self, training, *args, **kwargs):
        super(BNLSTMCell, self).__init__(*args, **kwargs)     
        self._training = training
       
    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        num_proj = self._num_units if self._num_proj is None else self._num_proj

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])


        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        with vs.variable_scope(scope or "lstm_cell", initializer=self._initializer) as unit_scope:             
            #concat_w = _get_concat_variable("W", [input_size.value + num_proj, 4 * self._num_units], dtype, self._num_unit_shards)
            
            W_xh = vs.get_variable("W_xh", shape=[input_size.value, 4 * self._num_units], dtype=dtype)
            W_hh = vs.get_variable("W_hh", shape=[self._num_units, 4 * self._num_units], dtype=dtype)
            xh = math_ops.matmul(inputs, W_xh)
            hh = math_ops.matmul(m_prev, W_hh)
            bn_xh = my_batch_norm(xh,self._training,recurrent=True)
            lstm_matrix = bn_xh + hh
            i, j, f, o = array_ops.split(1, 4, lstm_matrix)
            # Diagonal connections
            if self._use_peepholes:
                with vs.variable_scope(unit_scope) as projection_scope:
                    if self._num_unit_shards is not None:
                        projection_scope.set_partitioner(None)
                w_f_diag = vs.get_variable("w_f_diag", shape=[self._num_units], dtype=dtype)
                w_i_diag = vs.get_variable("w_i_diag", shape=[self._num_units], dtype=dtype)
                w_o_diag = vs.get_variable("w_o_diag", shape=[self._num_units], dtype=dtype)

            if self._use_peepholes:
                c = (sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev + sigmoid(i + w_i_diag * c_prev) * self._activation(j))
            else:
                c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * self._activation(j))

            if self._cell_clip is not None:
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

            if self._use_peepholes:
                m = sigmoid(o + w_o_diag * c) * self._activation(c)
            else:
                m = sigmoid(o) * self._activation(c)
            
            if self._num_proj is not None:
                with vs.variable_scope("projection") as proj_scope:
                    if self._num_proj_shards is not None:
                        proj_scope.set_partitioner(partitioned_variables.fixed_size_partitioner(self._num_proj_shards))
                    m = _linear(m, self._num_proj, bias=False, scope=scope)
                if self._proj_clip is not None:
                    m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else array_ops.concat_v2([c, m], 1))
        return m, new_state