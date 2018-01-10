import tensorflow as tf

import pandas as pd
from tensorflow.contrib.rnn import LSTMBlockCell, BasicLSTMCell, DropoutWrapper, MultiRNNCell

BASIC = 'basci'
BLOCK = 'block'
CUDNN = 'cudnn'




class MTBModel(object):
    def __init__(self, is_training, config, input_):
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None

        with tf.device('/gpu:0'):
            inputs = inputs

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        output, state = self._build_rnn_graph(inputs, config, is_training)

    def _build_rnn_graph(self, inputs, config, is_training):
        if config.rnn_mode == CUDNN:
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):

        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and config.keep_prob < 1:
                cell = DropoutWrapper(cell,output_keep_prob=config.keep_prob)
            return cell

        cell = MultiRNNCell([make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, date_type())

        state = self._initial_state

        outputs = []

        with tf.variable_scope('RNN'):
            for time_step in range(self.num_steps):
                if time_step > 0 : tf.get_variable_scope().resue_variables()
                (cell_output, state) = cell(inputs[:,time_step,:],state)
                outputs.append(cell_output)
            output = tf.reshape(tf.concat(outputs,1),[-1,config.hidden_size])
        return output, state


    def _get_lstm_cell(self, config, is_training):
        if config.rnn_mode == BASIC:
            return BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=not is_training)

        if config.rnn_mode == BLOCK:
            return LSTMBlockCell(config.hidden_size, forget_bias=0.0)

        raise ValueError('rnn_mode %s not supported' % config.rnn_mode)


def main():
    pass


if __name__ == '__main__':
    main()
