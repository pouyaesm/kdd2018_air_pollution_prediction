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
        self.has_future = True

        # Init configuration
        self.time_steps = cfg[const.TIME_STEPS]
        super(Hybrid, self).__init__(cfg, self.time_steps)
        self._fg = self._fg = HybridFG(cfg={
            const.CITY: cfg[const.CITY],
            const.FEATURE_DIR: cfg.get(const.FEATURE_DIR, ""),
            const.FEATURE: cfg[const.FEATURE],
            const.STATIONS: cfg[const.STATIONS],
            const.POLLUTANT: cfg[const.POLLUTANT],
            const.CHUNK_COUNT: cfg.get(const.CHUNK_COUNT, 1),
            const.TEST_FROM: cfg.get(const.TEST_FROM, ''),
            const.TEST_TO: cfg.get(const.TEST_TO, ''),
        }, time_steps=self.time_steps)
        # Path to save and restore the model
        self._model_path = self.config[const.MODEL_DIR] + \
                           self.config[const.FEATURE] + str(self.time_steps) + '_hybrid_#.mdl'

    @staticmethod
    def lstm(ts_x, time_steps, num_units, keep_prob_holder=None, scope='lstm'):
        """
            LSTM RNN with time-steps cells and num-units per cell
        :param keep_prob_holder:
        :param ts_x:
        :param time_steps:
        :param num_units:
        :param scope:
        :return:
        """
        with tf.name_scope(scope):
            ts_x_reshaped = tf.stack(tf.unstack(value=ts_x, num=time_steps, axis=1, name='input_steps'), axis=0)
            # rnn_cell = rnn.BasicLSTMCell(num_units, name=scope + '_cell')
            rnn_cell = rnn.LSTMCell(num_units, name=scope + '_cell')
            # randomly block outputs of time-steps to make the RNN robust
            # to noisy inputs (input_keep_prob has numerical instability)
            rnn_cell = rnn.DropoutWrapper(rnn_cell, output_keep_prob=keep_prob_holder)
            outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=ts_x_reshaped,
                                                     time_major=True, parallel_iterations=4, dtype="float32")
            lstm_kernel, lstm_bias = rnn_cell._cell.variables
            # lstm_kernel, lstm_bias = rnn_cell.variables
            tf.summary.histogram(scope + '_kernel', lstm_kernel)
            tf.summary.histogram(scope + '_bias', lstm_bias)

        return tf.transpose(outputs)[0]
        # return outputs[-1]

    @staticmethod
    def mlp(x, input_d, output_d, is_training_holder, scope='mlp'):
        with tf.name_scope(scope):
            # hidden layers
            layer = NN.batch_norm(input=x, is_training=is_training_holder, scope=scope + '_bn_1')
            layer = tf.nn.relu(NN.linear(input_d, output_d, 'hid1', input=layer))
            layer = NN.batch_norm(input=layer, is_training=is_training_holder, scope=scope + '_bn_2')
            # output layer
            layer = NN.linear(output_d, output_d, 'out', input=layer)
        return layer

    def build(self):
        cnx_x = tf.placeholder(tf.float32, (None, self._fg.get_context_count()), name='cnx_x')

        # lstm-s of weather time-series
        meo_x = dict()
        for name in self._fg.meo_keys:
            meo_x[name] = tf.placeholder(tf.float32, (None, self._fg.meo_steps, 1),
                                         name='ts_' + name + '_x')

        # lstm-s of weather time-series
        future_x = dict()
        for name in self._fg.future_keys:
            future_x[name] = tf.placeholder(tf.float32, (None, self._fg.future_steps, 1),
                                         name='ts_fut_' + name + '_x')

        # lstm-s of air quality time-series
        air_x = dict()
        for name in self._fg.air_keys:
            air_x[name] = tf.placeholder(tf.float32, (None, self._fg.air_steps, 1),
                                         name='ts_' + name + '_x')

        y = tf.placeholder(tf.float32, (None, 48), name='ts_y')

        keep_prob = tf.placeholder_with_default(1.0, (), name='dropout_keep_prob')
        is_training = tf.placeholder_with_default(False, (), name='is_training')
        learning_rate = tf.placeholder_with_default(0.05, (), name='learning_rate')

        mlp_input = list()  # input to the last NN that outputs the final prediction

        air_out = dict()
        for name, input in air_x.items():
            air_out[name] = Hybrid.lstm(input, self._fg.air_steps, 1, keep_prob, scope='lstm_' + name)
            mlp_input.append(air_out[name])

        mlp_d = self._fg.air_steps * len(air_out)  # input to mlp per air quality measure

        if self.has_meo:
            meo_out = dict()
            for name, input in meo_x.items():
                meo_out[name] = Hybrid.lstm(input, self._fg.meo_steps, 1, keep_prob, scope='lstm_' + name)
                mlp_input.append(meo_out[name])
            mlp_d += self._fg.meo_steps * len(meo_out)

        if self.has_future:
            future_out = dict()
            for name, input in future_x.items():
                future_out[name] = Hybrid.lstm(input, self._fg.future_steps, 1, keep_prob, scope='lstm_fut_' + name)
                mlp_input.append(future_out[name])
            mlp_d += self._fg.future_steps * len(future_out)

        if self.has_context:
            mlp_input.append(cnx_x)
            mlp_d += self._fg.get_context_count()

        mlp_x = tf.concat(mlp_input, axis=1, name='mlp_x')
        prediction = Hybrid.mlp(mlp_x, mlp_d, 48, is_training)

        # loss_function
        nom = tf.abs(tf.subtract(x=prediction, y=y))
        denom = tf.divide(x=tf.abs(prediction) + tf.abs(y), y=2)
        smape = tf.reduce_mean(tf.divide(x=nom, y=denom))

        tf.summary.scalar('SMAPE', smape)

        # mean absolute error or SMAPE for mean percent error
        loss_function = smape if self.config[const.LOSS_FUNCTION] == const.MEAN_PERCENT \
            else tf.reduce_mean(nom)
        # optimization (AdaGrad changes its learning rate during training)
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_function)

        # merge all summaries
        summary_all = tf.summary.merge_all()

        return {
            'cnx': cnx_x,
            'meo': meo_x,
            'future': future_x,
            'air': air_x,
            'keep_prob': keep_prob,
            'is_train': is_training,
            'lr': learning_rate,
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
        summary_writer = tf.summary.FileWriter('logs/hybrid/run2')
        summary_writer.add_graph(self._session.graph)

        # initialize session variables
        self._session.run(tf.global_variables_initializer())

        min_holdout_smp = 0.7  # lowest SMAPE error on hold-out set so far
        learning_rate = 0.05  # initial learning rate
        for i in range(0, epochs):
            context, meo, future, air, label = self._fg.next(batch_size=batch_size,
                                                           progress=i / epochs, rotate=rotate)
            # update the network
            learning_rate = 0.9995 * learning_rate
            self.run(model['train_step'], model=model, cnx=context, fut=future, meo=meo, air=air, lbl=label,
                                  kp=0.66, train=True, lr=learning_rate)
            # record network summary
            summary = self.run(model['summary'], model=model,
                     cnx=context, meo=meo, fut=future, air=air, lbl=label)
            summary_writer.add_summary(summary, i)
            if i % 10 == 0:
                train_smp = self.run(model['smape'], model=model,
                                     cnx=context, meo=meo, fut=future, air=air, lbl=label)
                context, meo, future, air, label = self._fg.holdout(key=const.VALID)
                valid_smp = self.run(model['smape'], model=model,
                                     cnx=context, meo=meo, fut=future, air=air, lbl=label)
                context, meo, future, air, label = self._fg.holdout(key=const.TEST)
                test_smp = self.run(model['smape'], model=model,
                                    cnx=context, meo=meo, fut=future, air=air, lbl=label)
                print(i, "SMAPE tr", train_smp, ", v", valid_smp, ", t", test_smp, "    lr", learning_rate)
                # If a 0.1% better model found according to hold set, save it
                if test_smp < min_holdout_smp - 0.001:
                    self.save_model(mode='best')
                    min_holdout_smp = test_smp

        self._model = model  # make model accessible to other methods

        # Report SMAPE error on test set
        context, meo, future, air, label = self._fg.holdout(key=const.TEST)
        print("Testing SMAPE:", self.run(model['smape'], model=model,
                                         cnx=context, meo=meo, fut=future, air=air, lbl=label))

        return self

    def run(self, nodes, model, cnx, meo, fut, air, lbl, kp=1.0, train=False, lr=None):
        """
            Run the mode for given computational nodes and inputs
        :param lr: learning rate
        :param train: is in training phase or not
        :param fut:
        :param kp: drop out keeping output probability
        :param nodes:
        :param model:
        :param cnx:
        :param meo:
        :param air:
        :param lbl:
        :return:
        """
        feed_dict = {model['cnx']: cnx, model['y']: lbl,
                     model['keep_prob']: kp, model['is_train']: train, model['lr']: lr}

        # feed each time series to a different lstm, instead of all to one lstm
        for i, name in enumerate(self._fg.meo_keys):
            feed_dict[model['meo'][name]] = meo[:, :, i:i + 1]  # input to lstm of 'name'

        for i, name in enumerate(self._fg.future_keys):
            feed_dict[model['future'][name]] = fut[:, :, i:i + 1]

        for i, name in enumerate(self._fg.air_keys):
            feed_dict[model['air'][name]] = air[:, :, i:i + 1]

        return self._session.run(nodes, feed_dict=feed_dict)

    def test(self):
        model = self._model
        context, meo_ts, future_ts, air_ts, label = self._fg.holdout(key=const.TEST)
        if len(context) > 0:
            test_smp = self.run(model['smape'], model=model,
                                cnx=context, meo=meo_ts, fut=future_ts, air=air_ts, lbl=label)
            station_count = len(self._fg._test[const.ID].unique())
            print("Testing SMAPE:", test_smp, 'for', station_count, 'stations')
            predicted_label = self.predict(x={'c': context, 'm': meo_ts, 'f': future_ts,
                                              'a': air_ts, 'l': label})
            self._fg.save_test(predicted_label)
        else:
            print("Empty hold-out set!")
        return self

    def predict(self, x):
        return self.run(self._model['predictor'],
                        model=self._model, cnx=x['c'], meo=x['m'], fut=x['f'], air=x['a'], lbl=x['l'])


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]
    cases = {
        'BJ': [
            # 'PM2.5',
            # 'PM10',
            # 'O3'
        ],
        'LD': [
            # 'PM2.5',
            'PM10'
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
                const.TEST_TO: '18-04-27 00',
                const.LOSS_FUNCTION: const.MEAN_PERCENT,
                const.CHUNK_COUNT: 10 if city == const.BJ else 4,
                const.ROTATE: 5,
                const.TIME_STEPS: 12,
                const.EPOCHS: 2500,
                const.BATCH_SIZE: 2000
            }
            hybrid = Hybrid(cfg).train().save_model()
            print(city, pollutant, "done!")
