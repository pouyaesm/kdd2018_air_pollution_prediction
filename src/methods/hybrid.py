"""
    Learn the time-series using an MLP over all stations all together
"""
import numpy as np
import pandas as pd
import const
import settings
import tensorflow as tf
from src import util
from tensorflow.contrib import rnn
from src.methods.lstm import LSTM
from src.methods.neural_net import NN


class Hybrid(LSTM):

    def __init__(self, cfg):
        self.time_steps = cfg[const.TIME_STEPS]
        super(Hybrid, self).__init__(cfg, self.time_steps)
        # Path to save and restore the model
        self._model_path = self.config[const.MODEL_DIR] + \
                           self.config[const.FEATURE] + str(self.time_steps) + '_rcnn.mdl'
        self.context_count = 1

    @staticmethod
    def lstm(ts_x, time_steps, num_units):
        with tf.name_scope("lstm"):
            ts_x_reshaped = tf.stack(tf.unstack(value=ts_x, num=time_steps, axis=1, name='input_steps'), axis=0)
            rnn_cell = rnn.BasicLSTMCell(num_units)
            outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=ts_x_reshaped,
                                                     time_major=True, parallel_iterations=4, dtype="float32")

        lstm_kernel, lstm_bias = rnn_cell.variables
        tf.summary.histogram('lstm_kernel', lstm_kernel)
        tf.summary.histogram('lstm_bias', lstm_bias)
        return outputs[-1]

    @staticmethod
    def mlp(x, input_d, output_d):
        with tf.name_scope("mlp"):
            hidden_d = int(input_d / 2)
            # hidden layers
            layer = tf.nn.relu(NN.linear(input_d, hidden_d, 'hid1', input=x))
            layer = NN.batch_norm(hidden_d, hidden_d, input=layer)
            # output layer
            layer = NN.linear(hidden_d, output_d, 'out', input=layer)
        return layer

    def build(self):
        num_units = 1
        d_output = 48
        d_input = 1
        lstm_out_d = 24

        cnx_x = tf.placeholder(tf.float32, (None, self.context_count), name='cnx_x')
        ts_x = tf.placeholder(tf.float32, (None, self.time_steps, 1), name='ts_x')
        y = tf.placeholder(tf.float32, (None, 48), name='ts_y')

        lstm_out = Hybrid.lstm(ts_x, self.time_steps, lstm_out_d)
        mlp_x = tf.concat([lstm_out, cnx_x], axis=1, name='mlp_x')
        prediction = Hybrid.mlp(mlp_x, lstm_out_d + self.context_count, 48)

        # loss_function
        nom = tf.abs(tf.subtract(x=prediction, y=y))
        denom = tf.divide(x=prediction + y, y=2)
        smape = tf.reduce_mean(tf.divide(x=nom, y=denom))
        # mean absolute error or SMAPE for mean percent error
        loss_function = smape if self.config[const.LOSS_FUNCTION] == const.MEAN_PERCENT \
            else tf.reduce_mean(nom)
        # optimization
        train_step = tf.train.AdamOptimizer(learning_rate=0.02).minimize(loss_function)

        # merge all summaries
        summary_all = tf.summary.merge_all()

        return {
            'cnx': cnx_x,
            'ts': ts_x,
            'y': y,
            'train_step': train_step,
            'loss': loss_function,
            'smape': smape,
            'summary': summary_all,
            'predictor': prediction
        }

    def train(self):
        batch_size = self.config[const.BATCH_SIZE]
        epochs = self.config[const.EPOCHS]

        model = self.build()

        # summary writer
        # summary_writer = tf.summary.FileWriter('logs/lstm/fast1')
        # summary_writer.add_graph(self.session.graph)

        # initialize session variables
        self.session.run(tf.global_variables_initializer())

        for i in range(0, epochs):
            batch_x, batch_y = self.fg.next(batch_size=batch_size, time_steps=self.time_steps)
            self.session.run(model['train_step'], feed_dict={model['x']: batch_x, model['y']: batch_y})
            # summary, _ = sess.run([summary_all, train_step], feed_dict={x: batch_x, y: batch_y})
            # summary_writer.add_summary(summary, i)
            if i % 10 == 0:
                smp, los = self.session.run([model['smape'], model['loss']],
                                    feed_dict={model['x']: batch_x, model['y']: batch_y})
                print(i, " Loss ", los, ", SMAPE ", smp)

        self.model = model  # make model accessible to other methods

        # Report SMAPE error on test set
        test_data, test_label = self.fg.test(time_steps=self.time_steps)
        print("Testing SMAPE:", self.session.run(model['smape'], feed_dict=
        {model['x']: test_data, model['y']: test_label}))

        return self

    def test(self):
        test_data, test_label = self.fg.test(time_steps=self.time_steps)
        smp = self.session.run(self.model['smape'], feed_dict=
        {self.model['x']: test_data, self.model['y']: test_label})
        print("Testing SMAPE:", smp)
        predicted_label = self.predict(test_data)
        self.fg.save_test(predicted_label)
        return self

    def predict(self, x):
        return self.session.run(self.model['predictor'], feed_dict={self.model['x']: x})


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]
    cases = {
        'BJ': [
            'PM2.5',
            # 'PM10',
            # 'O3'
        ],
        'LD': [
            # 'PM2.5',
            # 'PM10'
        ]
    }
    # For low values of pollutants MAE works better than SMAPE!
    # So for all pollutants of london and O3 of beijing we use MAE
    for city in cases:
        for pollutant in cases[city]:
            print(city, pollutant, "started..")
            cfg = {
                const.MODEL_DIR: config[const.MODEL_DIR],
                const.FEATURE_DIR: config[const.FEATURE_DIR],
                const.FEATURE: getattr(const, city + '_' + pollutant.replace('.', '') + '_'),
                const.LOSS_FUNCTION: const.MEAN_ABSOLUTE,
                const.EPOCHS: 1000,
                const.BATCH_SIZE: 500,
                const.TIME_STEPS: 48
            }
            hybrid = Hybrid(cfg).train().save_model()
            print(city, pollutant, "done!")
