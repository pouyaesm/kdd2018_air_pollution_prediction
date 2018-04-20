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


class LSTM:

    def __init__(self, cfg):
        self.config = cfg
        self.model = Sequential()

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

    def train(self):
        batch_size = 1000
        time_steps = 48
        fg = LSTMFG({
            const.FEATURES: self.config[const.FEATURES],
        }, input_hours=time_steps)
        x, y, loss, opt = self.build()
        # initialize variables
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for i in range(0, 3000):
            batch_x, batch_y = fg.next(batch_size=batch_size, time_steps=time_steps)
            sess.run(opt, feed_dict={x: batch_x, y: batch_y})
            if i % 10 == 0:
                los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
                print("For iter ", i)
                print("Loss ", los)
                print("__________________")

        test_data, test_label = fg.test(time_steps=time_steps)
        print("Testing SMAPE:", sess.run(loss, feed_dict={x: test_data, y: test_label}))
        return self

    def build(self):
        num_units = 1
        time_steps = 48
        n_input = 1
        # weights and biases of appropriate shape to accomplish above task
        out_weights = tf.Variable(tf.random_normal([time_steps, time_steps]))
        out_bias = tf.Variable(tf.random_normal([time_steps]))

        # defining placeholders
        # input image placeholder
        x = tf.placeholder("float", [None, time_steps, n_input])
        # input label placeholder
        y = tf.placeholder("float", [None, time_steps])

        # processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
        input = tf.unstack(x, time_steps, 1)

        # defining the network
        # lstm_layer = rnn.BasicLSTMCell(num_units)
        lstm_layer = rnn.LSTMCell(num_units)
        outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")

        # converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
        # prediction = tf.transpose(outputs)[0]  # tf.matmul(outputs[-1], out_weights) + out_bias
        prediction = tf.matmul(tf.transpose(outputs)[0], out_weights) + out_bias

        # loss_function
        nom = tf.abs(tf.subtract(x=prediction, y=y))
        denom = tf.divide(x=prediction + y, y=2)
        loss = tf.reduce_mean(tf.divide(x=nom, y=denom))
        # optimization
        opt = tf.train.AdamOptimizer(learning_rate=0.025).minimize(loss)

        # model evaluation
        return x, y, loss, opt

    def predict(self, x):
        return self.model.predict(x)

    @staticmethod
    def evaluate(actual, forecast):
        actual_array = actual.values.reshape(actual.size)
        forecast_array = np.array(forecast).reshape(forecast.size)
        return util.SMAPE(forecast=forecast_array, actual=actual_array)

    def load(self):
        self.model = tf.keras.models.load_model(filepath=self.config[const.MODEL])
        return self

    def save(self):
        self.model.save(filepath=self.config[const.MODEL])
        return self


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]
    lstm = LSTM({
        const.FEATURES: config[const.BJ_PM10_] + 'lstm_features.csv',
    })
    lstm.train()
    print("Done!")
