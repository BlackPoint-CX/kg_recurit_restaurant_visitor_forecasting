#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author(s) : BlackPoint-CX
# CreateTime : 2018/1/10 9:25

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMBlockCell, BasicLSTMCell, DropoutWrapper
from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(flag_name='model', default_value='small',
                    docstring='A type of model. Possible options are : small, medium,large.')
flags.DEFINE_string('data_path', None, 'Where the training/test data is stored.')
flags.DEFINE_string('save_path', None, 'Model output directory.')
flags.DEFINE_boolean('use_fp16', False, 'Train using 16-bit floats instead of 32bit floats.')
flags.DEFINE_integer('num_gpus', 1, "If larger than 1, Grappler AutoParallel optimizer will create multiple "
                                    "training replicas with each GPU  running one replica.")
flags.DEFINE_string('rnn_mode', None, "The low level implementation of lstm cell: one of CUDNN, "
                                      "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                                      "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = 'basic'
CUDNN = 'cudnn'
BLOCK = 'block'


def data_tyep():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class INPUT(object):
    """
    handle the process of inputting.
    """


class RNNModel(object):
    """
    RNNModel
    """

    def __int__(self):
        self._lr = tf.Varibale(0.0, trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=[], name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr)
        pass

    def _build_run_graph(self, inputs, config, is_training):
        if config.rnn_mode == CUDNN:
            return self._build_run_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_run_graph_lstm(inputs, config, is_training)

    def _build_run_graph_cudnn(inputs, config, is_training):
        pass

    def _get_lstm_cell(self, config, is_training):
        if config.rnn_mode == BASIC:
            return BasicLSTMCell(num_units=config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                                 reuse=not is_training)
        if config.rnn_mode == BLOCK:
            return LSTMBlockCell(num_units=config.hidden_size, forget_bias=0.0)
        raise ValueError('rnn_mode %s not supported' % config.rnn_mode)

    def _build_run_graph_lstm(self, inputs, config, is_training):
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and config.keep_prob < 1:
                cell = DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            return cell

        cell = MultiRNNCell([make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(config.batch_size, data_tyep())
        state = self._initial_state
        outputs = []

        with tf.variable_scope('RNN'):
            for time_stemp in range(self.num_steps):
                if time_stemp > 0: tf.get_variable_scope().reuse_variables()  # 这个是啥意思 为什么要复用?
                (cell_output, state) = cell(inputs, state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])  # 重变形 为什么还要和1进行concat ?
        return output, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


def get_config():
    """Get model config."""
    config = None
    # if FLAGS.model == "small":
    #     config = SmallConfig()
    config = SmallConfig()
    # if FLAGS.rnn_mode :
    #     config.rnn_mode = FLAGS.rnn_mode
    config.rnn_mode = FLAGS.rnn_mode
    return config

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


def rrvf_producer(raw_data,batch_size,num_steps,name=None):
    with tf.name_scope(name,'rrvf_producer',[raw_data,batch_size,num_steps]):
        raw_data = tf.convert_to_tensor(raw_data,name='raw_data',dtype=tf.float32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0:batch_size * batch_len],[batch_size,batch_len])

        epoch_size = (batch_len-1) // num_steps  # 为啥这里要减一

        assertion = tf.assert_positive(epoch_size, message='epoch_size == 0 ,decrease batch_size or num_steps')


    with tf.control_dependencies([assertion]): # control_dependencies什么意思
        epoch_size = tf.identity(epoch_size,name='epoch_size') # identity 什么用处

    i= tf.train.range_input_producer(limit=epoch_size,shuffle=False).dequeue()




def main():

    pass


if __name__ == '__main__':
    main()
