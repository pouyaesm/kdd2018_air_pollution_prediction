"""
    Learn the time-series using an MLP over all stations all together
"""
import numpy as np
import pandas as pd
import const
import settings
import tensorflow as tf
from src import util
from src.feature_generators.lstm_fg import LSTMFG
from keras.models import Sequential
from tensorflow.contrib import rnn
from src.methods.model import Model


class LSTM(Model):

    def __init__(self, cfg, time_steps):
        self.config = cfg
        self.model = Sequential()
        self.time_steps = time_steps
        # create tensorflow session
        self.session = tf.Session()
        self.model = dict()  # contains different points in a tensorflow computation graph
        self.fg = self.fg = LSTMFG({
                const.FEATURES: self.config[const.FEATURES],
        }, input_hours=self.time_steps)
        # Path to save and restore the model
        self._model_path = self.config[const.FEATURES] + str(self.time_steps) + '_lstm_model.mdl'

    @staticmethod
    def replace_time(data: pd.DataFrame):
        """
            Replace the complete datetime with its (day of week, hour)
        :param data:
        :return:
        """
        date = pd.to_datetime(data[const.TIME], format=const.T_FORMAT, utc=True)
        data.insert(loc=0, column='hour', value=date.dt.hour)
        data.insert(loc=0, column='dayofweek', value=(date.dt.dayofweek + 1) % 7)  # 0: monday -> 0: sunday
        data.drop(columns=[const.TIME], inplace=True)

    @staticmethod
    def drop_time_location(data: pd.DataFrame):
        """
        :param data:
        :return:
        """
        data.drop(columns=[const.LONG], inplace=True)
        data.drop(columns=[const.LAT], inplace=True)
        data.drop(columns=[const.TIME], inplace=True)

    def build(self):
        num_units = 1
        d_output = 48
        d_input = 1
        # weights and biases of appropriate shape to accomplish above task
        out_weights = tf.Variable(tf.random_normal([self.time_steps, d_output]), name='out_weights')
        out_bias = tf.Variable(tf.random_normal([d_output]), name='out_bias')

        # defining placeholders
        # input image placeholder
        x = tf.placeholder("float", [None, self.time_steps, d_input], name='x')
        # input label placeholder
        y = tf.placeholder("float", [None, d_output], name='y')

        # processing the input tensor from [batch_size,n_steps,n_input]
        # to "time_steps" number of [batch_size,n_input] tensors
        x_reshaped = tf.unstack(x, self.time_steps, 1, name='input_steps')

        # defining the network
        # lstm_layer = rnn.BasicLSTMCell(num_units, name='BasicLSTMCell')
        lstm_layer = rnn.LSTMCell(num_units, name='LSTMCell')
        outputs, _ = rnn.static_rnn(lstm_layer, x_reshaped, dtype="float32")

        # converting last output of dimension [batch_size,num_units]
        # to [batch_size,n_classes] by out_weight multiplication
        # prediction = tf.transpose(outputs)[0]  # tf.matmul(outputs[-1], out_weights) + out_bias
        # prediction = tf.matmul(outputs[-1], out_weights) + out_bias
        prediction = tf.matmul(tf.transpose(outputs)[0], out_weights) + out_bias

        # loss_function
        nom = tf.abs(tf.subtract(x=prediction, y=y))
        denom = tf.divide(x=prediction + y, y=2)
        smape = tf.reduce_mean(tf.divide(x=nom, y=denom))
        # mean absolute error or SMAPE for mean percent error
        loss_function = smape if self.config[const.LOSS_FUNCTION] == const.MEAN_PERCENT \
            else tf.reduce_mean(nom)
        # optimization
        train_step = tf.train.AdamOptimizer(learning_rate=0.025).minimize(loss_function)

        # summaries of interest
        lstm_kernel, lstm_bias = lstm_layer.variables
        tf.summary.scalar('SMAPE', smape)
        tf.summary.histogram('lstm_kernel', lstm_kernel)
        tf.summary.histogram('lstm_bias', lstm_bias)
        tf.summary.histogram('output_weights', out_weights)
        tf.summary.histogram('output_bias', out_bias)

        # merge all summaries
        summary_all = tf.summary.merge_all()

        return {
            'x': x,
            'y': y,
            'train_step': train_step,
            'loss': loss_function,
            'smape': smape,
            'summary': summary_all,
            'predictor': prediction
        }

    def train(self):
        batch_size = 1000

        model = self.build()

        # summary writer
        summary_writer = tf.summary.FileWriter('logs/lstm/fast1')
        summary_writer.add_graph(self.session.graph)

        # initialize session variables
        self.session.run(tf.global_variables_initializer())

        for i in range(0, 3000):
            batch_x, batch_y = self.fg.next(batch_size=batch_size, time_steps=self.time_steps)
            self.session.run(model['train_step'], feed_dict={model['x']: batch_x, model['y']: batch_y})
            # summary, _ = sess.run([summary_all, train_step], feed_dict={x: batch_x, y: batch_y})
            # summary_writer.add_summary(summary, i)
            if i % 10 == 0:
                smp, los = self.session.run([model['smape'], model['loss']],
                                    feed_dict={model['x']: batch_x, model['y']: batch_y})
                print(i, " Loss ", los, ", SMAPE ", smp)

        self.model = model  # make model accessible to other methods
        self.save_model()
        return self

    def test(self):
        test_data, test_label = self.fg.test(time_steps=self.time_steps)
        smp = self.session.run(self.model['smape'], feed_dict=
        {self.model['x']: test_data, self.model['y']: test_label})
        print("Testing SMAPE:", smp)
        predicted_label = self.predict(test_data)
        self.fg.save_test(predicted_label)
        return self

    def evaluate(self, actual, forecast):
        # return self.session.run(self.model['smape'],
        #                             feed_dict={self.model['x']: batch_x, model['y']: batch_y})
        actual_array = actual.reshape(actual.size)
        forecast_array = forecast.reshape(forecast.size)
        return util.SMAPE(forecast=forecast_array, actual=actual_array)

    def predict(self, x):
        return self.session.run(self.model['predictor'], feed_dict={self.model['x']: x})

    def load_model(self):
        self.model = self.build()
        tf.train.Saver().restore(sess=self.session, save_path=self._model_path)
        return self

    def save_model(self):
        save_path = tf.train.Saver().save(sess=self.session, save_path=self._model_path)
        print("Model saved in path: %s" % save_path)
        return self


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]
    pollutant = 'PM2.5'
    features_bj = config[getattr(const, 'BJ_' + pollutant.replace('.', '') + '_')]
    features_ld = config[getattr(const, 'LD_' + pollutant.replace('.', '') + '_')]
    # For low values of pollutants MAE works better than SMAPE!
    # So for all pollutants of london and O3 of beijing we use MAE
    config_bj = {
        const.FEATURES: features_bj,
        const.LOSS_FUNCTION: const.MEAN_ABSOLUTE
    }
    config_ld = {
        const.FEATURES: features_ld,
        const.LOSS_FUNCTION: const.MEAN_ABSOLUTE
    }
    lstm = LSTM(config_bj, time_steps=48).train()
    # lstm = LSTM(config_ld, time_steps=24).train()
    print("Done!")
