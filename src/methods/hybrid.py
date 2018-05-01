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
        self.has_air_long = True
        self.has_meo = True
        self.has_meo_long = True
        self.has_future = True

        # Init configuration
        super(Hybrid, self).__init__(cfg, -1)
        self._fg = self._fg = HybridFG(cfg=cfg)
        # Path to save and restore the model
        self._model_path = self.config[const.MODEL_DIR] + \
                           self.config[const.FEATURE] + self._fg.path_indicator + '_hybrid_#.mdl'

    @staticmethod
    def lstm(ts_in, time_steps, num_units, keep_prob_holder=None, scope='lstm'):
        """
            LSTM RNN with time-steps cells and num-units per cell
        :param keep_prob_holder:
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
            # input layer
            # batch normalizing aggregated inputs makes them comparable with 0.01-0.02 better SMAPE
            layer = NN.batch_norm(input=x, is_training=is_training_holder, scope=scope + '_bn_1')
            # hidden layer 1 (2 * output size)
            # going through a larger layer larger than output gives an effective
            # model capacity boost which is essential!
            # Deeper networks cannot be trained effectively here
            layer = tf.nn.relu(NN.linear(input_d, output_d, 'hid1', input=layer))
            # 1) Adding batch normalization between (last hidden and output) causes slow improvement
            #  and high accuracy instability among (train, valid, test) even after O(100) epochs
            # 2) More than one layer worsens best SMAPE 0.1
            # output layer
            layer = NN.linear(output_d, output_d, 'out', input=layer)
        return layer

    def build(self):
        cnx_in = tf.placeholder(tf.float32, (None, self._fg.get_context_count()), name='cnx_in')

        # lstm-s of weather time-series
        meo_in = dict()
        meo_long_in = dict()
        for name in self._fg.meo_keys:
            meo_in[name] = tf.placeholder(tf.float32, (None, self._fg.meo_steps, 1),
                                         name=name + '_in')
            meo_long_in[name] = tf.placeholder(tf.float32, (None, self._fg.meo_long_steps, 1),
                                               name=name + '_l_in')

        # lstm-s of weather time-series
        future_in = dict()
        for name in self._fg.future_keys:
            future_in[name] = tf.placeholder(tf.float32, (None, self._fg.future_steps, 1),
                                         name=name + '_f_in')

        # lstm-s of air quality time-series
        air_in = dict()
        air_long_in = dict()
        for name in self._fg.air_keys:
            air_in[name] = tf.placeholder(tf.float32, (None, self._fg.air_steps, 1),
                                         name=name + '_in')
            air_long_in[name] = tf.placeholder(tf.float32, (None, self._fg.air_long_steps, 1),
                                               name=name + '_l_in')

        y = tf.placeholder(tf.float32, (None, 48), name='out')

        keep_prob = tf.placeholder_with_default(1.0, (), name='dropout_keep_prob')
        is_training = tf.placeholder_with_default(False, (), name='is_training')
        learning_rate = tf.placeholder_with_default(0.05, (), name='learning_rate')

        mlp_input = list()  # input to the last NN that outputs the final prediction

        # lstm addition function for different time series
        def add_lstm(ts_in, steps, name_prefix=''):
            ts_out = dict()
            for name, input in ts_in.items():
                ts_out[name] = Hybrid.lstm(input, steps, 1, keep_prob, scope='lstm_' + name_prefix + name)
                mlp_input.append(ts_out[name])
            return steps * len(ts_out)

        mlp_d = add_lstm(ts_in=air_in, steps=self._fg.air_steps)  # input to mlp per air quality measure
        if self.has_air_long:
            mlp_d += add_lstm(ts_in=air_long_in, steps=self._fg.air_long_steps, name_prefix='l_')
        if self.has_meo:
            mlp_d += add_lstm(ts_in=meo_in, steps=self._fg.meo_steps)
        if self.has_meo_long:
            mlp_d += add_lstm(ts_in=meo_long_in, steps=self._fg.meo_long_steps, name_prefix='l_')
        if self.has_future:
            mlp_d += add_lstm(ts_in=future_in, steps=self._fg.future_steps, name_prefix='f_')
        if self.has_context:
            mlp_input.append(cnx_in)
            mlp_d += self._fg.get_context_count()

        mlp_in = tf.concat(mlp_input, axis=1, name='mlp_in')
        prediction = Hybrid.mlp(mlp_in, mlp_d, 48, is_training)

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
            'cnx': cnx_in,
            'meo': meo_in,
            'meo_long': meo_long_in,
            'future': future_in,
            'air': air_in,
            'air_long': air_long_in,
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

        min_valid_smp = 0.7  # lowest SMAPE error on hold-out set so far
        learning_rate = 0.05  # initial learning rate
        for i in range(0, epochs):
            train = self._fg.next(batch_size=batch_size, progress=i / epochs, rotate=rotate)
            # update the network
            learning_rate = 0.9995 * learning_rate
            # SMAPE(dropout = 0.5) - SMAPE(dropout = 0.66) ~ 0.07 !
            # Therefore, with dropout = 0.5, output is deprived of critical input information
            # Dropout = 0.75 tends toward over-fitting less chance to find good local
            self.run(model['train_step'], model=model, x=train, kp=0.66, train=True, lr=learning_rate)
            # record network summary
            summary = self.run(model['summary'], model=model, x=train)
            summary_writer.add_summary(summary, i)
            if i % 10 == 0:
                train_smp = self.run(model['smape'], model=model, x=train)
                valid = self._fg.holdout(key=const.VALID)
                test = self._fg.holdout(key=const.TEST)
                valid_smp = self.run(model['smape'], model=model, x=valid)
                test_smp = self.run(model['smape'], model=model, x=test)
                weighted_smp = (valid_smp * len(valid) + test_smp * len(test)) / (len(valid) + len(test))
                print(i, "SMAPE tr", train_smp, ", v", valid_smp, ", t", test_smp, ", w", weighted_smp,
                      "    lr", learning_rate)
                # If a 0.1% better model found according to hold set, save it
                # Also validation and test set better not differ much indicating generalization
                if weighted_smp < min_valid_smp - 0.001 and test_smp - valid_smp < 0.05:
                    self.save_model(mode='best')
                    min_valid_smp = weighted_smp

        self._model = model  # make model accessible to other methods

        # Report SMAPE error on test set
        test = self._fg.holdout(key=const.TEST)
        print("Testing SMAPE:", self.run(model['smape'], model=model, x=test))

        return self

    def run(self, nodes, model, x, kp=1.0, train=False, lr=None):
        """
            Run the mode for given computational nodes and inputs
        :param x: dictionary of values for model placeholders
        :param lr: learning rate
        :param train: is in training phase or not
        :param kp: drop out keeping output probability
        :param nodes:
        :param model:
        :return:
        """
        feed_dict = {model['cnx']: x['c'], model['y']: x['l'],
                     model['keep_prob']: kp, model['is_train']: train, model['lr']: lr}

        # feed each time series to a different lstm, instead of all to one lstm
        for i, name in enumerate(self._fg.meo_keys):
            feed_dict[model['meo'][name]] = x['m'][:, :, i:i + 1]  # input to lstm of 'name'
            feed_dict[model['meo_long'][name]] = x['ml'][:, :, i:i + 1]

        for i, name in enumerate(self._fg.future_keys):
            feed_dict[model['future'][name]] = x['f'][:, :, i:i + 1]

        for i, name in enumerate(self._fg.air_keys):
            feed_dict[model['air'][name]] = x['a'][:, :, i:i + 1]
            feed_dict[model['air_long'][name]] = x['al'][:, :, i:i + 1]

        return self._session.run(nodes, feed_dict=feed_dict)

    def test(self):
        model = self._model
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
        return self.run(self._model['predictor'], model=self._model, x=x)


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]
    cases = {
        'BJ': [
            'PM2.5',
            'PM10',
            'O3'
        ],
        'LD': [
            'PM2.5',
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
                const.TRAIN_FROM: '00-01-01 00',
                const.TRAIN_TO: '18-03-30 00',
                const.VALID_FROM: '18-04-01 23',
                const.VALID_TO: '18-04-15 00',
                const.TEST_FROM: '18-04-15 23',
                const.TEST_TO: '18-04-29 00',
                const.LOSS_FUNCTION: const.MEAN_PERCENT,
                const.ROTATE: 5,
                const.EPOCHS: 2500,
                # Batch size 2.5K may land in a bad optima, 5k also may be bad
                # but 4k is considerably robust!
                const.BATCH_SIZE: 4000
            }
            cfg.update(HybridFG.get_size_config(city=city))  # configuration of feature sizes
            hybrid = Hybrid(cfg).train().save_model()
            print(city, pollutant, "done!")
