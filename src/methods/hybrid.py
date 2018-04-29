"""
    Learn the time-series using an MLP over all stations all together
"""
import const
import settings
import tensorflow as tf
from tensorflow.contrib import rnn
from src.methods.lstm import LSTM
from src.methods.neural_net import NN
from src.feature_generators.hybrid_fg import HybridFG


class Hybrid(LSTM):

    def __init__(self, cfg):
        # Structure parameters
        self.has_context = True
        self.has_meo = True

        # Init configuration
        self.time_steps = cfg[const.TIME_STEPS]
        super(Hybrid, self).__init__(cfg, self.time_steps)
        self._fg = self._fg = HybridFG(cfg={
            const.CITY: cfg[const.CITY],
            const.FEATURE_DIR: cfg[const.FEATURE_DIR],
            const.FEATURE: cfg[const.FEATURE],
            const.STATIONS: cfg[const.STATIONS],
            const.POLLUTANT: cfg[const.POLLUTANT],
            const.CHUNK_COUNT: cfg.get(const.CHUNK_COUNT, 1),
            const.TEST_FROM: cfg[const.TEST_FROM],
            const.TEST_TO: cfg[const.TEST_TO],
        }, time_steps=self.time_steps)
        # Path to save and restore the model
        self._model_path = self.config[const.MODEL_DIR] + \
                           self.config[const.FEATURE] + str(self.time_steps) + '_hybrid.mdl'

    @staticmethod
    def lstm(ts_x, time_steps, num_units, scope='lstm'):
        with tf.name_scope(scope):
            ts_x_reshaped = tf.stack(tf.unstack(value=ts_x, num=time_steps, axis=1, name='input_steps'), axis=0)
            # rnn_cell = rnn.BasicLSTMCell(num_units, name=scope + '_cell')
            rnn_cell = rnn.LSTMCell(num_units, name=scope + '_cell')
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
            # hidden_d = input_d  # int(input_d / 2)
            # hidden layers
            layer = NN.batch_norm(input_d, input_d, input=x)
            layer = tf.nn.relu(NN.linear(input_d, output_d, 'hid1', input=layer))
            layer = NN.batch_norm(output_d, output_d, input=layer)
            # layer = tf.nn.relu(NN.linear(output_d, output_d, 'hid2', input=layer))
            # layer = NN.batch_norm(output_d, output_d, input=layer)
            # output layer
            layer = NN.linear(output_d, output_d, 'out', input=layer)
        return layer

    def build(self):
        meo_out_d = 6
        air_out_d = 18

        cnx_x = tf.placeholder(tf.float32, (None, self._fg.get_context_count()), name='cnx_x')

        # lstm-s of weather time-series
        meo_x = dict()
        for name in self._fg.meo_keys:
            meo_x[name] = tf.placeholder(tf.float32, (None, self._fg.meo_steps, 1),
                                         name='ts_' + name + '_x')

        # lstm-s of air quality time-series
        air_x = dict()
        # for name in [const.PM25]:
        # for name in [const.PM25, const.PM10]:
        for name in self._fg.air_keys:
            air_x[name] = tf.placeholder(tf.float32, (None, self._fg.air_steps, 1),
                                         name='ts_' + name + '_x')

        y = tf.placeholder(tf.float32, (None, 48), name='ts_y')

        # meo_out = Hybrid.lstm(ts_meo_x, self.fg.meo_steps, meo_out_d, scope='lstm_meo_')
        # air_out = Hybrid.lstm(ts_air_x, self.fg.air_steps, air_out_d, scope='lstm_air_')
        # mlp_x = tf.concat([meo_out, air_out, cnx_x], axis=1, name='mlp_x')
        # prediction = Hybrid.mlp(mlp_x, meo_out_d + air_out_d + self.fg.get_context_count(), 48)

        mlp_input = list()  # input to the last NN that outputs the final prediction

        air_out = dict()
        for name, input in air_x.items():
            air_out[name] = Hybrid.lstm(input, self._fg.air_steps, 1, scope='lstm_' + name)
            mlp_input.append(air_out[name])

        mlp_d = self._fg.air_steps * len(air_out)  # input to mlp per air quality measure

        if self.has_meo:
            meo_out = dict()
            for name, input in meo_x.items():
                meo_out[name] = Hybrid.lstm(input, self._fg.meo_steps, 1, scope='lstm_' + name)
                mlp_input.append(meo_out[name])
            mlp_d += self._fg.meo_steps * len(meo_out)

        if self.has_context:
            mlp_input.append(cnx_x)
            mlp_d += self._fg.get_context_count()

        mlp_x = tf.concat(mlp_input, axis=1, name='mlp_x')
        prediction = Hybrid.mlp(mlp_x, mlp_d, 48)

        # loss_function (normalized)
        # mean, std = FG.get_statistics(
        #     name=self.config[const.POLLUTANT], city=self.config[const.CITY])
        # prediction_real = prediction * std + mean
        # y_real = y * std + mean
        # nom = tf.abs(tf.subtract(x=prediction_real, y=y_real))
        # denom = tf.divide(x=tf.abs(prediction_real) + tf.abs(y_real), y=2)
        # smape = tf.reduce_mean(tf.divide(x=nom, y=denom))

        # loss_function
        nom = tf.abs(tf.subtract(x=prediction, y=y))
        denom = tf.divide(x=tf.abs(prediction) + tf.abs(y), y=2)
        smape = tf.reduce_mean(tf.divide(x=nom, y=denom))

        # mean absolute error or SMAPE for mean percent error
        loss_function = smape if self.config[const.LOSS_FUNCTION] == const.MEAN_PERCENT \
            else tf.reduce_mean(nom)
        # optimization
        train_step = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss_function)

        # merge all summaries
        summary_all = tf.summary.merge_all()

        return {
            'cnx': cnx_x,
            'meo': meo_x,
            'air': air_x,
            'y': y,
            'train_step': train_step,
            'loss': loss_function,
            'smape': smape,
            'summary': summary_all,
            'predictor': prediction
        }

    def train(self):
        batch_size = self.config[const.BATCH_SIZE]  # data point for each gradient descent
        epochs = self.config[const.EPOCHS]  # number of gradient descent iterations
        rotate = self.config[const.ROTATE]  # number of iterations over whole data during a complete epoch
        model = self.build()

        # summary writer
        summary_writer = tf.summary.FileWriter('logs/hybrid/run1')
        summary_writer.add_graph(self._session.graph)

        # initialize session variables
        self._session.run(tf.global_variables_initializer())

        for i in range(0, epochs):
            context, meo_ts, air_ts, label = self._fg.next(batch_size=batch_size,
                                                           progress=i / epochs, rotate=rotate)
            # run(model['train_step'], model=model, c=context, m=meo_ts, a=air_ts, l=label)
            summary, _ = self.run([model['summary'], model['train_step']],
                                  model=model, c=context, m=meo_ts, a=air_ts, l=label)
            summary_writer.add_summary(summary, i)
            if i % 10 == 0:
                train_smp = self.run(model['smape'], model=model,
                               c=context, m=meo_ts, a=air_ts, l=label)
                context, meo_ts, air_ts, label = self._fg.holdout(key=const.TEST)
                test_smp = self.run(model['smape'], model=model,
                               c=context, m=meo_ts, a=air_ts, l=label)
                print(i, "SMAPE tr", train_smp, " vs ", test_smp, "tst")

        self._model = model  # make model accessible to other methods

        # Report SMAPE error on test set
        context, meo_ts, air_ts, label = self._fg.holdout(key=const.TEST)
        print("Testing SMAPE:", self.run(model['smape'], model=model,
                                    c=context, m=meo_ts, a=air_ts, l=label))

        return self

    def run(self, nodes, model, c, m, a, l):
        """
            Run the mode for given computational nodes and inputs
        :param nodes:
        :param model:
        :param c:
        :param m:
        :param a:
        :param l:
        :return:
        """
        feed_dict = {model['cnx']: c, model['y']: l}
        # feed each time series to a different lstm, instead of all to one lstm
        air_order = {const.PM25: 0, const.PM10: 1, const.O3: 2}
        for name, input in model['air'].items():
            i = air_order[name]
            feed_dict[input] = a[:, :, i:i + 1]  # input to lstm of 'name'

        meo_order = {const.TEMP: 0, const.HUM: 1, const.WSPD: 2}
        for name, input in model['meo'].items():
            i = meo_order[name]
            feed_dict[input] = m[:, :, i:i + 1]  # input to lstm of 'name'

        return self._session.run(nodes, feed_dict=feed_dict)

    def test(self):
        model = self._model
        context, meo_ts, air_ts, label = self._fg.holdout(key=const.TEST)
        test_smp = self.run(model['smape'], model=model,
                            c=context, m=meo_ts, a=air_ts, l=label)
        print("Testing SMAPE:", test_smp)
        predicted_label = self.predict(x={'c': context, 'm': meo_ts, 'a': air_ts, 'l': label})
        self._fg.save_test(predicted_label)
        return self

    def predict(self, x):
        return self.run(self._model['predictor'],
                        model=self._model, c=x['c'], m=x['m'], a=x['a'], l=x['l'])


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
                const.CITY: city,
                const.MODEL_DIR: config[const.MODEL_DIR],
                const.FEATURE_DIR: config[const.FEATURE_DIR],
                const.POLLUTANT: pollutant,
                const.FEATURE: getattr(const, city + '_' + pollutant.replace('.', '') + '_'),
                const.STATIONS: config[getattr(const, city + '_STATIONS')],
                const.TEST_FROM: '18-04-01 23',
                const.TEST_TO: '18-04-26 23',
                const.LOSS_FUNCTION: const.MEAN_PERCENT,
                const.CHUNK_COUNT: 8,
                const.TIME_STEPS: 12,
                const.EPOCHS: 10000,
                const.ROTATE: 2,
                const.BATCH_SIZE: 2500
            }
            hybrid = Hybrid(cfg).train().save_model()
            print(city, pollutant, "done!")
