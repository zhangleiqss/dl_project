import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.rnn_cell import RNNCell,LSTMCell,LSTMStateTuple


#summary for scalar
def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name.replace(':', '_'), mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name.replace(':', '_'), stddev)
        tf.summary.scalar('max/' + name.replace(':', '_'), tf.reduce_max(var))
        tf.summary.scalar('min/' + name.replace(':', '_'), tf.reduce_min(var))
        tf.summary.histogram(name.replace(':', '_'), var)

def my_minimize_loss(opt, loss, params, clip_type=None, max_clip_grad=1.0, dependency=None):
    grads = opt.compute_gradients(loss, params)
    #Processing gradients before applying them
    if clip_type == 'clip_value':
        capped_gvs = [(None if grad is None else tf.clip_by_value(grad, -max_clip_grad, max_clip_grad), var) for grad, var in grads] 
    elif clip_type == 'clip_norm':
        capped_gvs = [(None if grad is None else tf.clip_by_norm(grad, max_clip_grad), var) for grad, var in grads] 
    elif clip_type == 'clip_global_norm':
        capped_gvs, grads_norm = tf.clip_by_global_norm([grad for grad, var in grads], max_clip_grad)
        capped_gvs = list(zip(capped_gvs, params))
    else:
        capped_gvs = grads
    if dependency == None:
        optim = opt.apply_gradients(capped_gvs)
    else:
        with tf.control_dependencies([dependency]):
            optim = opt.apply_gradients(capped_gvs)
    return optim, grads, capped_gvs

############################################################################################
#the seq loss for the targets which have variable length
#y_pred shape: [None, maxlen, nb_classes]    y_true shape: [None, maxlen]
def variable_seq_loss(y_pred, y_true):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, y_true)
    mask = tf.to_float(tf.sign(tf.abs(y_true)))
    cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
    seq_loss = tf.reduce_mean(cross_entropy,0)
    return seq_loss

#the accuracy for the targets which have variable length
def varibale_accuracy(y_pred, y_true):
    pred_idx = tf.to_int32(tf.argmax(y_pred, 2))
    accuracy_flag = tf.to_float(tf.equal(pred_idx, y_true))
    mask = tf.to_float(tf.sign(tf.abs(y_true)))
    accuracy_flag *= mask
    accuracy = tf.reduce_sum(accuracy_flag)/tf.reduce_sum(mask) 
    return accuracy

def varibale_topk_accuracy(y_pred, y_true, k):
    y_pred_new = tf.reshape(y_pred, [-1, y_pred.get_shape()[2].value])
    y_true_new = tf.reshape(y_true, [-1])
    accuracy_flag = tf.to_float(tf.nn.in_top_k(y_pred_new,y_true_new,k=k))
    mask = tf.to_float(tf.sign(tf.abs(y_true_new)))
    accuracy_flag *= mask
    tok_accuracy = tf.reduce_sum(accuracy_flag)/tf.reduce_sum(mask) 
    return tok_accuracy

def variable_set_accuracy_1d(y_pred, y_true):
    mask = tf.not_equal(tf.zeros_like(y_true), y_true)
    diff = tf.listdiff(tf.boolean_mask(y_pred,mask),tf.boolean_mask(y_true,mask))
    accuracy = 1 - tf.reduce_sum(tf.to_float(tf.ones_like(diff.idx)))/tf.reduce_sum(tf.to_float(mask))
    return accuracy

def variable_set_accuracy(y_pred, y_true, batch_size):
    pred_idx = tf.to_int32(tf.argmax(y_pred, 2))
    #y_pred_list = tf.unpack(pred_idx, axis=0)
    #y_true_list = tf.unpack(y_true, axis=0)
    accuracy = tf.reduce_mean([variable_set_accuracy_1d(pred_idx[i,:], y_true[i,:]) for i in range(batch_size)])
    return accuracy

def variable_topk_set_accuracy(y_pred, y_true, batch_size):
    _, pred_idx = tf.nn.top_k(y_pred, k=10, sorted=True, name=None)
    accuracy = tf.reduce_mean([variable_set_accuracy_1d(pred_idx[i,:], y_true[i,:]) for i in range(batch_size)])
    return accuracy

def reset_nan(input_tensor, number=1e-5):
    input_tensor = tf.select(tf.is_nan(input_tensor), tf.ones_like(input_tensor) * 1e-5, input_tensor)
    return input_tensor

def print_varaible_shape_summary():
    for var in tf.trainable_variables():
        print var.name, var.get_shape() 
        
##########################################################################################################
def _get_fans(shape):
    """Returns values of input dimension and output dimension, given `shape`.
    
    Args:
      shape: A list of integers.
    
    Returns:
      fan_in: An int. The value of input dimension.
      fan_out: An int. The value of output dimension.
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        kernel_size = np.prod(shape[:2])
        fan_in = shape[-2] * kernel_size
        fan_out = shape[-1] * kernel_size
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def he_uniform(name, shape, scale=1, dtype=tf.float32):
    """See He et al. 2015 `http://arxiv.org/pdf/1502.01852v1.pdf`
    """
    fin, _ = _get_fans(shape)
    s = np.sqrt(1. * scale / fin)
    shape = shape if isinstance(shape, (tuple, list)) else [shape]
    W = tf.Variable(tf.random_uniform(shape, minval=-s, maxval=s),dtype=dtype,name=name)
    #w = tf.get_variable(name, shape, dtype=dtype,initializer=tf.random_uniform_initializer(minval=-s, maxval=s))
    return W

########################################################################################################
class ZoneoutWrapper(LSTMCell):
    def __init__(self, cell, training, state_out_prob=0.0, cell_out_prob=0.0, seed=None):
        if not isinstance(cell, LSTMCell):
            raise TypeError("The parameter cell is not a RNNCell.")
        if (isinstance(state_out_prob, float) and
            not (state_out_prob >= 0.0 and state_out_prob <= 1.0)):
            raise ValueError("Parameter state_out_prob must be between 0 and 1: %d"
                       % state_out_prob)
        if (isinstance(cell_out_prob, float) and
            not (cell_out_prob >= 0.0 and cell_out_prob <= 1.0)):
            raise ValueError("Parameter cell_out_prob must be between 0 and 1: %d"
                       % cell_out_prob)
        self._cell = cell
        self._state_out_prob = state_out_prob
        self._cell_out_prob = cell_out_prob
        self._seed = seed
        self._training = training
    
    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        output, new_state = self._cell(inputs, state, scope)
        def train():
            cell_update = nn_ops.dropout(state[0], self._cell_out_prob, seed=self._seed) + nn_ops.dropout(new_state[0], 1-self._cell_out_prob, seed=self._seed)
            state_update = nn_ops.dropout(state[1], self._state_out_prob, seed=self._seed) + nn_ops.dropout(new_state[1], 1-self._state_out_prob, seed=self._seed)
            return cell_update, state_update
        
        def test():
            cell_update = state[0] * self._cell_out_prob + new_state[0] * (1-self._cell_out_prob)
            state_update = state[1] * self._state_out_prob + new_state[1] * (1-self._state_out_prob)
            return cell_update, state_update

        cell_update, state_update = tf.cond(self._training,train,test)
        new_state_update = LSTMStateTuple(cell_update,state_update)      
        return output, new_state_update
