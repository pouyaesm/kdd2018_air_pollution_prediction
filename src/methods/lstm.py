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

    def __init__(self, cfg, time_steps):
        self.config = cfg
        self.model = Sequential()
        self.time_steps = time_steps

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
        fg = LSTMFG({
            const.FEATURES: self.config[const.FEATURES] +
                            str(self.time_steps) + '_lstm_features.csv',
        }, input_hours=self.time_steps)

        x, y, train_step, loss_function, smape, summary_all = self.build()

        # initialize variables
        init = tf.global_variables_initializer()
        sess = tf.Session()

        # summary writer
        summary_writer = tf.summary.FileWriter('logs/lstm/fast1')
        summary_writer.add_graph(sess.graph)

        sess.run(init)
        for i in range(0, 3000):
            batch_x, batch_y = fg.next(batch_size=batch_size, time_steps=self.time_steps)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
            # summary, _ = sess.run([summary_all, train_step], feed_dict={x: batch_x, y: batch_y})
            # summary_writer.add_summary(summary, i)
            if i % 10 == 0:
                smp, los = sess.run([smape, loss_function], feed_dict={x: batch_x, y: batch_y})
                print(i, " Loss ", los, ", SMAPE ", smp)

        test_data, test_label = fg.test(time_steps=self.time_steps)
        print("Testing SMAPE:", sess.run(smape, feed_dict={x: test_data, y: test_label}))
        return self

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
        lstm_input = tf.unstack(x, self.time_steps, 1, name='input_steps')

        # defining the network
        # lstm_layer = rnn.BasicLSTMCell(num_units)
        lstm_layer = rnn.LSTMCell(num_units, name='LSTMCell')
        outputs, _ = rnn.static_rnn(lstm_layer, lstm_input, dtype="float32")

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

        # model evaluation
        return x, y, train_step, loss_function, smape, summary_all

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
    pollutant = 'PM10'
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
    lstm = LSTM(config_bj, time_steps=1).train()
    # lstm = LSTM(config_ld).train()
    print("Done!")
