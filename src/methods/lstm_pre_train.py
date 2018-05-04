"""
    Learn the time-series using an MLP over all stations all together
"""
import const
import settings
import tensorflow as tf
from src.methods.hybrid_base import HybridBase
from src.feature_generators.hybrid_fg import HybridFG


class LSTMPre(HybridBase):

    def __init__(self, cfg):
        # Init configuration
        super(LSTMPre, self).__init__(cfg)
        self.scope = self.config[const.POLLUTANT]
        # Path to save and restore the model
        self.model_path = self.config[const.MODEL_DIR] + \
                          self._fg.feature_indicator + "\\" + self._fg.param_indicator + '_pre_#.mdl'

    def build_model(self):

        # lstm-s of air quality time-series
        ts_in = tf.placeholder(tf.float32, (None, self._fg.air_steps, 1),
                       name=self._fg.pollutant + '_pre_in')

        y = tf.placeholder(tf.float32, (None, 48), name='out')

        learning_rate = tf.placeholder_with_default(0.05, (), name='learning_rate')

        # time series of the pollutant directly used for output prediction
        prediction_direct = self.lstm(ts_in, self._fg.air_steps, 1, transform_d=48,
                                      scope='lstm_pre_' + self._fg.pollutant)

        # optimization (AdaGrad changes its learning rate during training)
        loss = self.loss_function_absolute(prediction_direct, actual=y, prefix='direct_')
        smape = self.loss_function_smape(prediction_direct, actual=y, prefix='direct_')
        train_pre = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(loss)

        # merge all summaries
        summary_all = tf.summary.merge_all()

        return {
            'pre_in': ts_in,
            'lr': learning_rate,
            'y': y,
            'smape_direct': smape,
            'train_direct': train_pre,
            'summary': summary_all,
            'predictor': prediction_direct
        }

    def train_model(self):

        model = self.model

        batch_size = self.config[const.BATCH_SIZE]  # data point for each gradient descent
        epochs = self.config[const.EPOCHS]  # number of gradient descent iterations
        rotate = self.config[const.ROTATE]  # number of iterations over whole data during a complete epoch

        summary_writer = self.initialize_summary_writer(scope='pre-train')

        self.session.run(tf.global_variables_initializer())  # initialize graph variables

        learning_rate = 0.025  # initial learning rate
        for i in range(0, epochs):
            train = self._fg.next(batch_size=batch_size, progress=i / epochs, rotate=rotate)
            # update the network
            learning_rate = 0.9995 * learning_rate
            self.run(model['train_direct'], model=model, x=train,
                     train=True, direct=True, lr=learning_rate)
            # record network summary
            summary = self.run(model['summary'], model=model, x=train)
            summary_writer.add_summary(summary, i)
            if i % 10 == 0:
                train_smp = self.run(model['smape_direct'], model=model, x=train)
                valid = self._fg.holdout(key=const.VALID)
                test = self._fg.holdout(key=const.TEST)
                valid_smp = self.run(model['smape_direct'], model=model, x=valid)
                test_smp = self.run(model['smape_direct'], model=model, x=test)
                weighted_smp = (valid_smp * len(valid) + test_smp * len(test)) / (len(valid) + len(test))
                print(i, "SMAPE tr", train_smp, ", v", valid_smp, ", t", test_smp, ", w", weighted_smp,
                      "    lr", learning_rate)
        self.save_model()
        return self

    def run(self, nodes, model, x, kp=1.0, train=False, direct=False, lr=None):
        """
            Run the mode for given computational nodes and inputs
        :return:
        """
        i = self._fg.air_keys.index(self._fg.pollutant)  # index of pollutant data
        feed_dict = {model['pre_in']: x['a'][:, :, i:i + 1], model['y']: x['l'], model['lr']: lr}

        return self.session.run(nodes, feed_dict=feed_dict)


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
                const.ROTATE: 3,
                const.EPOCHS: 1500,
                const.BATCH_SIZE: 1500
            }
            cfg.update(HybridFG.get_size_config(city=city, key='05-02'))  # configuration of feature sizes
            hybrid = LSTMPre(cfg).build().train()
            print(city, pollutant, "done!")
