import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell,LSTMCell,LSTMStateTuple
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops


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
            i, j, f, o = array_ops.split(lstm_matrix, 4, 1)
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