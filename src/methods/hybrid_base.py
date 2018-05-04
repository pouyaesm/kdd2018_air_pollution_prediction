"""
    Learn the time-series using an MLP over all stations all together
"""
import const
import tensorflow as tf
from tensorflow.contrib import rnn
from src.methods.lstm import LSTM
from src.methods.neural_net import NN
import os.path
from src.feature_generators.hybrid_fg import HybridFG


class HybridBase(LSTM):

    def __init__(self, cfg):
        # Structure parameters
        self.has_context = True
        self.has_air_long = True
        self.has_meo = True
        self.has_meo_long = True
        self.has_future = True

        # Init configuration
        super(HybridBase, self).__init__(cfg, -1)
        self._fg = self._fg = HybridFG(cfg=cfg)

    @staticmethod
    def loss_function_smape(predict, actual, prefix=''):
        # loss_function_smape
        nom = tf.abs(tf.subtract(x=predict, y=actual))
        denom = tf.divide(x=tf.abs(predict) + tf.abs(actual), y=2)
        smape = tf.reduce_mean(tf.divide(x=nom, y=denom))
        tf.summary.scalar(prefix + 'SMAPE', smape)
        # # mean absolute error or SMAPE for mean percent error
        # loss = smape if self.config[const.LOSS_FUNCTION] == const.MEAN_PERCENT \
        #     else tf.reduce_mean(nom)
        return smape

    @staticmethod
    def loss_function_absolute(predict, actual, prefix=''):
        # loss_function_smape
        error = tf.reduce_mean(tf.abs(tf.subtract(x=predict, y=actual)))
        tf.summary.scalar(prefix + 'MAE', error)
        return error

    @staticmethod
    def lstm(ts_in, time_steps, num_units, transform_d=0, has_dropout=True, keep_prob=1.0, scope='lstm'):
        """
            LSTM RNN with time-steps cells and num-units per cell
        :param has_dropout:
        :param transform_d: dimension to linearly transform the output of LSTM
        :param keep_prob:
        :param ts_in:
        :param time_steps:
        :param num_units:
        :param scope:
        :return:
        """
        with tf.name_scope(scope):
            ts_x_reshaped = tf.stack(tf.unstack(value=ts_in, num=time_steps, axis=1, name='input_steps'), axis=0)
            # rnn_cell = rnn.BasicLSTMCell(num_units, name=scope + '_cell')
            rnn_cell = rnn.LSTMCell(num_units, name=scope + '_cell')
            # Randomly block outputs of time-steps to make the RNN robust
            # to noisy inputs and increase regularization (highly effective)
            # input_keep_prob or output_keep_prob are not effective!
            if has_dropout:
                rnn_cell = rnn.DropoutWrapper(rnn_cell, output_keep_prob=keep_prob)
            outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=ts_x_reshaped,
                                                     time_major=True, parallel_iterations=4, dtype="float32")
            if transform_d > 0:
                out_weights = tf.Variable(tf.random_normal([time_steps, transform_d]), name='out_weights')
                out_bias = tf.Variable(tf.random_normal([transform_d]), name='out_bias')
                outputs = tf.matmul(tf.transpose(outputs)[0], out_weights) + out_bias
                tf.summary.histogram('out_weights', out_weights)
                tf.summary.histogram('out_bias', out_bias)
            else:
                outputs = tf.transpose(outputs)[0]
            lstm_kernel, lstm_bias = rnn_cell._cell.variables if has_dropout else rnn_cell.variables
            tf.summary.histogram('kernel', lstm_kernel)
            tf.summary.histogram('bias', lstm_bias)
            outputs = tf.identity(outputs, name='output')
        return outputs
        # return outputs[-1]

    @staticmethod
    def default_layer(input_d, output_d, input, is_training, scope,
                      keep_prob=1.0, postfix='', has_dropout=True, has_batch=True):
        with tf.name_scope(scope):
            if has_dropout:
                layer = tf.contrib.layers.dropout(
                    inputs=input, keep_prob=keep_prob, is_training=is_training, scope='dp' + postfix)
            else:
                layer = input
            layer = NN.linear(input_d, output_d, 'hid' + postfix, input=layer, add_summary=True)
            layer = NN.batch_norm(input=layer, is_training=is_training, scope='bn' + postfix) \
                if has_batch else layer
        return layer

    @staticmethod
    def merger(x, input_d, output_d, keep_prob, is_training,
               has_dropout=True, has_batch=True, multiplier=2, layer_count=1, scope='mlp'):
        with tf.name_scope(scope):
            # batch normalizing aggregated inputs makes them comparable, suitable to be summed
            layer = NN.batch_norm(input=x, is_training=is_training, scope=scope + '_bn_0')
            coefficient = multiplier ** layer_count
            for i in range(1, layer_count + 1):
                in_d = input_d if i == 1 else int(output_d * coefficient)
                coefficient /= multiplier
                out_d = int(output_d * coefficient)
                layer = tf.nn.relu(HybridBase.default_layer(in_d, out_d, layer, is_training,
                                             scope, keep_prob, '_' + str(i),
                                                        has_dropout=has_dropout, has_batch=has_batch))
            return layer

    @staticmethod
    def head(x, input_d, output_d, keep_prob, is_training, scope='mlp'):
        with tf.name_scope(scope):
            # 1) Adding batch normalization between (last hidden and output) caused slow improvement
            #  and accuracy instability for train and / or test data
            # 2) Also adding dropout at last layers causes performance decline
            layer = HybridBase.merger(x, input_d, output_d, keep_prob, is_training,
                                  layer_count=1, scope=scope, has_dropout=False, has_batch=False)

            # output layer
            layer = NN.linear(output_d, output_d, 'out', input=layer)
        return layer

    def run(self, nodes, model, x, kp=1.0, train=False, direct=False, lr=None):
        """
            Run the mode for given computational nodes and inputs
        :param direct: if false no gradient flows into the LSTM responsible for direct prediction
            of next 48 hours from pollutant previous values
        :param x: dictionary of values for model placeholders
        :param lr: learning rate
        :param train: is in training phase or not
        :param kp: drop out keeping output probability
        :param nodes:
        :param model:
        :return:
        :rtype: tensorflow.Tensor
        """
        return tf.Tensor

    def initialize_summary_writer(self, scope='default'):
        # summary writer
        # find first available folder to put the results in
        run = 0
        while True:
            run += 1
            if os.path.isdir('logs/%s/run%d' % (scope, run)) is not True:
                break
        summary_writer = tf.summary.FileWriter('logs/%s/run%d' % (scope, run))
        summary_writer.add_graph(self.session.graph)
        return summary_writer

    def test(self):
        model = self.model
        test = self._fg.holdout(key=const.TEST)
        if len(test['c']) > 0:
            test_smp = self.run(model['smape'], model=model, x=test)
            station_count = len(self._fg._test[const.ID].unique())
            print("Testing SMAPE:", test_smp, 'for', station_count, 'stations')
            predicted_label = self.predict(x=test)
            self._fg.save_test(predicted_label)
        else:
            print("Empty hold-out set!")
        return self

    def predict(self, x):
        return self.run(self.model['predictor'], model=self.model, x=x)
