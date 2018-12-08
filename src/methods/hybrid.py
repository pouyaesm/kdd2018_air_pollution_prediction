"""
    Learn the time-series using an MLP over all stations all together
"""
import tensorflow as tf
import numpy as np
import const
import settings
from src.methods.hybrid_base import HybridBase
from src.methods.lstm_pre_train import LSTMPre
from src.feature_generators.hybrid_fg import HybridFG


class Hybrid(HybridBase):

    def __init__(self, cfg):
        # Init configuration
        super(Hybrid, self).__init__(cfg)
        self.scope = 'hybrid_' + cfg[const.POLLUTANT]
        # Path to save and restore the model
        self.model_path = self.config[const.MODEL_DIR] + \
                          self._fg.feature_indicator + "\\" + self._fg.param_indicator + '_hybrid_#.mdl'
        # pre-trained models to be used as input to the hybrid model
        self.pre_trained = dict()

    def build_model(self):
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

        # lstm addition function for different time series
        def add_lstm(ts_in, list: list, steps, name_prefix=''):
            for name, input in ts_in.items():
                list.append(Hybrid.lstm(input, steps, 1,
                                           has_dropout=False, keep_prob=keep_prob,
                                          scope='lstm_' + name_prefix + name))
            return steps * len(ts_in)

        # input to mlp per air quality measure
        head_input = list()
        merge_input = list()
        head_d = merge_d = 0
        with tf.variable_scope(self.scope + '_lstm'):
            merge_d = add_lstm(air_in, merge_input, self._fg.air_steps)
            if self.has_air_long:
                merge_d += add_lstm(air_long_in, merge_input, self._fg.air_long_steps, 'l_')
            if self.has_meo:
                merge_d += add_lstm(meo_in, merge_input, self._fg.meo_steps)
            if self.has_meo_long:
                merge_d += add_lstm(meo_long_in, merge_input, self._fg.meo_long_steps, 'l_')
            # put future weather time series closer to end
            # due to their prediction power
            if self.has_future:
                head_d += add_lstm(future_in, head_input, self._fg.future_steps, 'f_')
                # merge_d += add_lstm(ts_in=future_in, steps=self._fg.future_steps, name_prefix='f_')

        # output of all LSTMs are merged
        # using an MLP and its output is squeezed into a merge_out_d vector
        merge_in = tf.concat(merge_input, axis=1, name='merge_in')
        merge_out_d = int(merge_d / 2)
        with tf.variable_scope(self.scope + '_merger'):
            merge_out = Hybrid.merger(merge_in, merge_d, merge_out_d, keep_prob,
                                      is_training, layer_count=1, has_batch=False, scope='merger')

        # time series of the pollutant directly used for output prediction
        # prediction_direct = Hybrid.lstm(air_in[self.pollutant], self._fg.air_steps, 1,
        #
        head_input.append(merge_out)
        head_d += merge_out_d
        cfg = self.config.copy()
        for pollutant in self._fg.air_keys:
            cfg.update({const.POLLUTANT: pollutant})
            # load the model into current graph, and load its weights into current session
            lstm_pre = LSTMPre(cfg).load_model(session=self.session, model_scope=pollutant)
            # Feed the pollutant placeholder to that its pre-trained predictor
            # Get the output of pre-trained graph
            prediction = lstm_pre.get_model()['predictor']
            # Pre-trained graph is kept to fed required data to its placeholders
            self.pre_trained[pollutant] = lstm_pre
            # Do not update weights of LSTM that is trained to predict the output directly
            # during the mixed (tostal) training phase,
            # that is consider the output of this LSTM as a constant input to the mixed network
            prediction = tf.stop_gradient(prediction, name='%s_stop_direct' % pollutant)
            # map un-related pollutant predictions to a lower 'out_d' dimension
            if self._fg.pollutant != pollutant:
                embed_d = 6
                with tf.variable_scope(self.scope + '_embed_' + pollutant):
                    prediction = tf.nn.relu(self.default_layer(48, embed_d, prediction, is_training,
                                                    keep_prob=keep_prob, scope='embed', has_batch=False))
                head_d += embed_d
            else:
                head_d += 48
            head_input.append(prediction)

        # merge context with merged output of all time series
        if self.has_context:
            head_input.append(cnx_in)
            head_d += self._fg.get_context_count()
        # work with the main graph since default graph is changed due to loading pre-trained models
        head_in = tf.concat(head_input, axis=1, name='head_in')
        with tf.variable_scope(self.scope + '_out'):
            prediction_mixed = Hybrid.head(head_in, head_d, 48, keep_prob, is_training)
            loss = self.loss_function_smape(prediction_mixed, actual=y)
            train_step = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(loss)
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
            'smape': loss,
            'train': train_step,
            'summary': summary_all,
            'predictor': prediction_mixed
        }

    def train_model(self):

        model = self.model

        batch_size = self.config[const.BATCH_SIZE]  # data point for each gradient descent
        epochs = self.config[const.EPOCHS]  # number of gradient descent iterations
        rotate = self.config[const.ROTATE]  # number of iterations over whole data during a complete epoch
        dropout_keep_prob = self.config.get(const.DROPOUT, 1.0)  # probability of keeping an LSTM output

        # summary writer
        summary_writer = self.initialize_summary_writer(scope='hybrid')

        # only initialize hybrid-related local/global variables
        # to avoid resetting weights of pre-trained models
        var_list = {v for v in tf.global_variables() if v.name.startswith(self.scope)}
        self.session.run(tf.variables_initializer(var_list))
        # self.session.run(tf.global_variables_initializer())

        min_valid_smp = 0.7  # lowest SMAPE error on hold-out set so far
        learning_rate = 0.05  # initial learning rate
        for i in range(0, epochs):
            train = self._fg.next(batch_size=batch_size, progress=i / epochs, rotate=rotate)
            # update the network
            learning_rate = 0.9995 * learning_rate
            self.run(model['train'], model=model, x=train, kp=dropout_keep_prob, train=True, lr=learning_rate)
            # record network summary
            # summary = self.run(model['summary'], model=model, x=train)
            # summary_writer.add_summary(summary, i)
            if i % 10 == 0:
                train_smp = self.run(model['smape'], model=model, x=train)
                valid = self._fg.holdout(key=const.VALID)
                test = self._fg.holdout(key=const.TEST)
                valid_smp = self.run(model['smape'], model=model, x=valid)
                test_smp = self.run(model['smape'], model=model, x=test)
                weighted_smp = (valid_smp * len(valid['c']) + test_smp * len(test['c'])) / (
                        len(valid['c']) + len(test['c']))
                print(i, "SMAPE tr", train_smp, ", v", valid_smp, ", t", test_smp, ", w", weighted_smp,
                      "    lr", learning_rate)
                # If a 0.1% better model found according to hold set, save it
                # Also validation and test set better not differ much indicating generalization
                if weighted_smp < min_valid_smp - 0.001 and test_smp - valid_smp < 0.05:
                    self.save_model(mode='best')
                    min_valid_smp = weighted_smp

        # Report SMAPE error on test set
        test = self._fg.holdout(key=const.TEST)
        print("Testing SMAPE:", self.run(model['smape'], model=model, x=test))

        return self

    def run(self, nodes, model, x, kp=1.0, train=False, direct=False, lr=None):
        feed_dict = {model['cnx']: x['c'], model['y']: x['l'],
                     model['keep_prob']: kp, model['lr']: lr,
                     model['is_train']: train}

        # feed each time series to a different lstm, instead of all to one lstm
        for i, name in enumerate(self._fg.meo_keys):
            feed_dict[model['meo'][name]] = x['m'][:, :, i:i + 1]  # input to lstm of 'name'
            feed_dict[model['meo_long'][name]] = x['ml'][:, :, i:i + 1]

        for i, name in enumerate(self._fg.future_keys):
            feed_dict[model['future'][name]] = x['f'][:, :, i:i + 1]

        for i, name in enumerate(self._fg.air_keys):
            feed_dict[model['air'][name]] = x['a'][:, :, i:i + 1]
            feed_dict[model['air_long'][name]] = x['al'][:, :, i:i + 1]
            # Assign the same input to pre-trained predictor of this pollutant
            pre_trained_in = self.pre_trained[name].get_model()['pre_in']
            feed_dict[pre_trained_in] = feed_dict[model['air'][name]]
            # Assign the label (actual values) to all pre_trained (useless)
            pre_trained_out = self.pre_trained[name].get_model()['y']
            feed_dict[pre_trained_out] = x['l']

        return self.session.run(nodes, feed_dict=feed_dict)


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]
    cases = {
        'BJ': [
            # 'PM2.5',
            # 'PM10',
            # 'O3'
        ],
        'LD': [
            'PM2.5',
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
                const.STATIONS: config[getattr(const, city + '_STATIONS')],
                const.TRAIN_FROM: '00-01-01 00',
                const.TRAIN_TO: '18-03-30 00',
                const.VALID_FROM: '18-04-01 23',
                const.VALID_TO: '18-04-15 00',
                const.TEST_FROM: '18-04-15 23',
                const.TEST_TO: '18-04-29 00',
                const.LOSS_FUNCTION: const.MEAN_PERCENT,
                # The value depends on data volume and model parameters
                # For more parameters and less data, lower keep-probability is better
                # Change it until get a similar accuracy for train and test
                const.DROPOUT: 0.66,
                # more rotates prevents the model from over-fitting on sub-data
                const.ROTATE: 6 if city == const.BJ else 15,
                const.EPOCHS: 500,
                const.BATCH_SIZE: 3000
            }
            cfg.update(HybridFG.get_size_config(city=city, key='05-02'))  # configuration of feature sizes
            hybrid = Hybrid(cfg).build().train().save_model()
            print(city, pollutant, "done!")
