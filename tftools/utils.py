import numpy as np
import shutil
import os
import tensorflow as tf


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

def test_edit_distance():
    graph = tf.Graph()
    with graph.as_default():
        truth = tf.sparse_placeholder(tf.int32)
        hyp = tf.sparse_placeholder(tf.int32)
        editDist = tf.edit_distance(hyp, truth, normalize=False)

    with tf.Session(graph=graph) as session:
        truthTest = sparse_tensor_feed([[0,1,2], [0,1,2,3,4]])
        hypTest = sparse_tensor_feed([[3,4,5], [0,1,2,2]])
        feedDict = {truth: truthTest, hyp: hypTest}
        dist = session.run([editDist], feed_dict=feedDict)
        print(dist)


##########################################################################################################
def model_logger_dir_prepare(logs_dir, model_dir, current_name):
    if os.path.exists(logs_dir + current_name):
        shutil.rmtree(logs_dir  + current_name)
    os.makedirs(logs_dir  + current_name)
    if os.path.exists(model_dir + current_name):
        shutil.rmtree(model_dir  + current_name)
    os.makedirs(model_dir + current_name)

