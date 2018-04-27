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
from src.feature_generators.hybrid_fg import HybridFG


class Hybrid(LSTM):

    def __init__(self, cfg):
        self.time_steps = cfg[const.TIME_STEPS]
        super(Hybrid, self).__init__(cfg, self.time_steps)
        self.fg = self.fg = HybridFG({
            const.FEATURE_DIR: self.config[const.FEATURE_DIR],
            const.FEATURE: self.config[const.FEATURE],
            const.STATIONS: self.config[const.STATIONS],
            const.POLLUTANT: self.config[const.POLLUTANT],
            const.CHUNK_COUNT: self.config[const.CHUNK_COUNT],
            const.TEST_FROM: self.config[const.TEST_FROM],
            const.TEST_TO: self.config[const.TEST_TO],
        }, time_steps=self.time_steps)
        # Path to save and restore the model
        self._model_path = self.config[const.MODEL_DIR] + \
                           self.config[const.FEATURE] + str(self.time_steps) + '_rcnn.mdl'
        # Structure parameters
        self.has_context = False
        self.has_meo = True
        self.context_count = 1


    @staticmethod
    def lstm(ts_x, time_steps, num_units, scope='lstm'):
        with tf.name_scope(scope):
            ts_x_reshaped = tf.stack(tf.unstack(value=ts_x, num=time_steps, axis=1, name='input_steps'), axis=0)
            rnn_cell = rnn.BasicLSTMCell(num_units, name=scope + '_cell')
            # rnn_cell = rnn.LSTMCell(num_units, name=scope + '_cell')
            outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=ts_x_reshaped,
                                                     time_major=True, parallel_iterations=4, dtype="float32")
            lstm_kernel, lstm_bias = rnn_cell.variables
            tf.summary.histogram(scope + '_kernel', lstm_kernel)
            tf.summary.histogram(scope + '_bias', lstm_bias)

        return tf.transpose(outputs)[0]
        # return outputs[-1]

    @staticmethod
    def mlp(x, input_d, output_d, scope='mlp'):
        with tf.name_scope(scope):
            hidden_d = input_d  # int(input_d / 2)
            # hidden layers
            layer = tf.nn.relu(NN.linear(input_d, hidden_d, 'hid1', input=x))
            layer = NN.batch_norm(hidden_d, hidden_d, input=layer)
            # output layer
            layer = NN.linear(hidden_d, output_d, 'out', input=layer)
        return layer

    def build(self):
        meo_out_d = 6
        air_out_d = 18
        meo_in_d = len(self.fg.meo_keys)  # number of features related to meteorology
        air_in_d = len(self.fg.air_keys)  # number of features related to air quality
        cnx_x = tf.placeholder(tf.float32, (None, self.fg.get_context_count()), name='cnx_x')
        ts_meo_x = tf.placeholder(tf.float32, (None, self.fg.meo_steps, meo_in_d), name='ts_meo_x')
        ts_air_x = tf.placeholder(tf.float32, (None, self.fg.air_steps, air_in_d), name='ts_air_x')
        y = tf.placeholder(tf.float32, (None, 48), name='ts_y')

        # meo_out = Hybrid.lstm(ts_meo_x, self.fg.meo_steps, meo_out_d, scope='lstm_meo_')
        # air_out = Hybrid.lstm(ts_air_x, self.fg.air_steps, air_out_d, scope='lstm_air_')
        # mlp_x = tf.concat([meo_out, air_out, cnx_x], axis=1, name='mlp_x')
        # prediction = Hybrid.mlp(mlp_x, meo_out_d + air_out_d + self.fg.get_context_count(), 48)

        meo_out = Hybrid.lstm(ts_meo_x, self.fg.meo_steps, 1, scope='lstm_meo_')
        air_out = Hybrid.lstm(ts_air_x, self.fg.air_steps, 1, scope='lstm_air_')

        mlp_input = [air_out]
        mlp_d = self.fg.air_steps
        if self.has_context:
            mlp_input.append(cnx_x)
            mlp_d += self.fg.get_context_count()
        if self.has_meo:
            mlp_input.append(meo_out)
            mlp_d += self.fg.meo_steps
        mlp_x = tf.concat(mlp_input, axis=1, name='mlp_x')
        prediction = Hybrid.mlp(mlp_x, mlp_d, 48)

        # loss_function
        nom = tf.abs(tf.subtract(x=prediction, y=y))
        denom = tf.divide(x=tf.abs(prediction) + y, y=2)
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
            'meo_ts': ts_meo_x,
            'air_ts': ts_air_x,
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
        summary_writer = tf.summary.FileWriter('logs/hybrid/run1')
        summary_writer.add_graph(self.session.graph)

        def run(nodes, model, c, m, a, l):
            return self.session.run(nodes, feed_dict={model['cnx']: c, model['meo_ts']: m,
                                        model['air_ts']: a, model['y']: l})

        # initialize session variables
        self.session.run(tf.global_variables_initializer())

        for i in range(0, epochs):
            context, meo_ts, air_ts, label = self.fg.next(batch_size=batch_size,
                                                          progress=i / epochs, rotate=2)
            # run(model['train_step'], model=model, c=context, m=meo_ts, a=air_ts, l=label)
            summary, _ = run([model['summary'], model['train_step']],
                                  model=model, c=context, m=meo_ts, a=air_ts, l=label)
            summary_writer.add_summary(summary, i)
            if i % 10 == 0:
                train_smp, train_loss = run([model['smape'], model['loss']], model=model,
                               c=context, m=meo_ts, a=air_ts, l=label)
                context, meo_ts, air_ts, label = self.fg.holdout(key=const.TEST)
                test_smp = run(model['smape'], model=model,
                               c=context, m=meo_ts, a=air_ts, l=label)
                print(i, " Loss ", train_loss, ", SMAPE tr", train_smp, " vs ", test_smp, "tst")

        self.model = model  # make model accessible to other methods

        # Report SMAPE error on test set
        context, meo_ts, air_ts, label = self.fg.holdout(key=const.TEST)
        print("Testing SMAPE:", run(model['smape'], model=model,
                                    c=context, m=meo_ts, a=air_ts, l=label))

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
                const.POLLUTANT: pollutant,
                const.FEATURE: getattr(const, city + '_' + pollutant.replace('.', '') + '_'),
                const.STATIONS: config[getattr(const, city + '_STATIONS')],
                const.TEST_FROM: '18-04-01 23',
                const.TEST_TO: '18-04-30 23',
                const.LOSS_FUNCTION: const.MEAN_PERCENT,
                const.CHUNK_COUNT: 4,
                const.EPOCHS: 3000,
                const.BATCH_SIZE: 1000,
                const.TIME_STEPS: 48
            }
            hybrid = Hybrid(cfg).train().save_model()
            print(city, pollutant, "done!")
