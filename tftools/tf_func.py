import numpy as np
import tensorflow as tf
import shutil
import os
from sklearn import metrics
from sklearn.metrics import roc_auc_score,f1_score
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell,LSTMCell,LSTMStateTuple
#from tensorflow.python.ops.rnn_cell import RNNCell,LSTMCell,LSTMStateTuple
import h5py

###########################################################################
def get_new_variable_scope(name, scope=None, reuse=False):
    if reuse==True and scope==None:
        scope = name
    vscope = tf.variable_scope(scope, default_name=name, reuse=reuse)
    return vscope
    
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

        
def my_compute_grad(opt, loss, params, clip_type=None, max_clip_grad=1.0):
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
    return grads, capped_gvs
        
def my_minimize_loss(opt, loss, params, clip_type=None, max_clip_grad=1.0, dependency=None):
    grads, capped_gvs = my_compute_grad(opt, loss, params, clip_type, max_clip_grad)
    if dependency == None:
        optim = opt.apply_gradients(capped_gvs)
    else:
        with tf.control_dependencies([dependency]):
            optim = opt.apply_gradients(capped_gvs)
    return optim, grads, capped_gvs

################     loss and accuracy      ###############################################
#the seq loss for the targets which have variable length
#y_pred shape: [None, maxlen, nb_classes]    y_true shape: [None, maxlen]
def variable_seq_loss(y_pred, y_true, lm_flag=False):
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, y_true)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    mask = tf.to_float(tf.sign(tf.abs(y_true)))
    if lm_flag:
        mask = tf.concat([tf.ones_like(mask[:,0:1]),tf.slice(mask,[0,0],[-1,mask.shape[-1].value-1])],1)
    cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)
    seq_loss = tf.reduce_mean(cross_entropy,0)
    return seq_loss

#the accuracy for the targets which have variable length
def varibale_accuracy(y_pred, y_true, lm_flag=False):
    pred_idx = tf.to_int32(tf.argmax(y_pred, 2))
    accuracy_flag = tf.to_float(tf.equal(pred_idx, y_true))
    mask = tf.to_float(tf.sign(tf.abs(y_true)))
    if lm_flag:
        mask = tf.concat([tf.ones_like(mask[:,0:1]),tf.slice(mask,[0,0],[-1,mask.shape[-1].value-1])],1)
    accuracy_flag *= mask
    accuracy = tf.reduce_sum(accuracy_flag)/tf.reduce_sum(mask) 
    return accuracy

def varibale_topk_accuracy(y_pred, y_true, k, lm_flag=False):
    y_pred_new = tf.reshape(y_pred, [-1, y_pred.get_shape()[2].value])
    y_true_new = tf.reshape(y_true, [-1])
    accuracy_flag = tf.to_float(tf.nn.in_top_k(y_pred_new,y_true_new,k=k))
    mask = tf.to_float(tf.sign(tf.abs(y_true)))
    if lm_flag:
        mask = tf.concat([tf.ones_like(mask[:,0:1]),tf.slice(mask,[0,0],[-1,mask.shape[-1].value-1])],1)
    mask = tf.reshape(mask, [-1])
    accuracy_flag *= mask
    tok_accuracy = tf.reduce_sum(accuracy_flag)/tf.reduce_sum(mask) 
    return tok_accuracy

def variable_set_accuracy_1d(y_pred, y_true):
    mask = tf.not_equal(tf.zeros_like(y_true), y_true)
    diff = tf.setdiff1d(tf.boolean_mask(y_pred,mask),tf.boolean_mask(y_true,mask))   
    #diff = tf.listdiff(tf.boolean_mask(y_pred,mask),tf.boolean_mask(y_true,mask))
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
        print(var.name, var.get_shape()) 
        
#######################################################################################################
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
    vscope = get_new_variable_scope('he_uniform')
    with vscope as scope:
        #print scope.name
        W = tf.get_variable(name, shape, dtype=dtype,initializer=tf.random_uniform_initializer(minval=-s, maxval=s))
        return W

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.         
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        #grad = tf.concat(0, grads)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
        #average_grads = [(None if grad is None else tf.clip_by_norm(grad, max_grad_norm), var) for grad, var in average_grads]                     
    return average_grads

###################################################################################################
def target_list_to_sparse_tensor(targetList):
    '''make tensorflow SparseTensor from list of targets, with each element
       in the list being a list or array with the values of the target sequence
       (e.g., the integer values of a character map for an ASR target string)
       See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
       for example of SparseTensor format'''
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices), np.array(vals), np.array(shape))

def model_logger_dir_prepare(logs_dir, model_dir, current_name):
    if os.path.exists(logs_dir + current_name):
        shutil.rmtree(logs_dir  + current_name)
    os.makedirs(logs_dir  + current_name)
    if os.path.exists(model_dir + current_name):
        shutil.rmtree(model_dir  + current_name)
    os.makedirs(model_dir + current_name)

def best_fscore(y_true, pred, labels=[1]):
    return np.max([f1_score(y_true, pred>x, labels=labels) for x in np.arange(0.001, 1, 0.001)])
    