"""
    Learn the time-series using an MLP over all stations all together
"""
import numpy as np
import pandas as pd
import const
import settings
import tensorflow as tf
from src import util
from src.preprocess import times
from src.methods.neural_net import NN
from keras.models import Sequential


class MLP:

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

    def build(self):
        # load_model data
        # stations = pd.read_csv(self.config[const.STATIONS], sep=";", low_memory=False)
        ts = pd.read_csv(self.config[const.FEATURE], sep=";", low_memory=False)

        # train = times.select(df=ts, time_key=const.TIME, from_time='00-01-01 00', to_time='17-11-31 23')
        # valid = times.select(df=ts, time_key=const.TIME, from_time='17-11-31 00', to_time='17-12-31 23')
        # test = times.select(df=ts, time_key=const.TIME, from_time='17-12-31 00', to_time='18-01-31 23')

        train = times.select(df=ts, time_key=const.TIME, from_time='00-01-01 00', to_time='17-12-31 23')
        valid = times.select(df=ts, time_key=const.TIME, from_time='17-12-31 23', to_time='17-12-31 23')
        test = times.select(df=ts, time_key=const.TIME, from_time='18-01-01 00', to_time='18-01-31 23')

        # MLP.replace_time(train)
        # MLP.replace_time(valid)
        # MLP.replace_time(test)

        train.drop(columns=[const.ID, const.TIME], inplace=True)
        valid.drop(columns=[const.ID, const.TIME], inplace=True)
        test.drop(columns=[const.ID, const.TIME], inplace=True)

        # pollutants = ['PM10']  # ['PM2.5', 'PM10', 'O3']
        # columns = ['forecast', 'actual', 'station', 'pollutant']
        # predictions = pd.DataFrame(data={}, columns=columns)
        feature_count = train.columns.size - 48
        x = range(0, feature_count)
        y = range(feature_count, feature_count + 48)
        x_train = train.iloc[:, x]
        y_train = train.iloc[:, y]
        x_valid = valid.iloc[:, x]
        y_valid = valid.iloc[:, y]
        x_test = test.iloc[:, x]
        y_test = test.iloc[:, y]

        self.model = NN.keras_mlp(x_train=x_train, y_train=y_train,
                                  x_valid=x_valid, y_valid=y_valid,
                                  loss=self.config[const.LOSS_FUNCTION])

        y_predict = self.predict(x_test)
        print('SMAPE:', MLP.evaluate(actual=y_test, forecast=y_predict))

        return self

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
    base_bj = config[const.BJ_PM10_]
    base_ld = config[const.LD_PM10_]
    config_bj = {
        const.STATIONS: config[const.BJ_STATIONS],
        const.FEATURE: base_bj + "mlp_features.csv",
        const.MODEL: base_bj + "mlp_model.mdl",
        const.LOSS_FUNCTION: const.MEAN_PERCENT
    }
    config_ld = {
        const.STATIONS: config[const.LD_STATIONS],
        const.FEATURE: base_ld + "mlp_features.csv",
        const.MODEL: base_ld + "mlp_model.mdl",
        const.LOSS_FUNCTION: const.MEAN_ABSOLUTE
    }
    # keras_mlp = MLP(config_bj).build().save_features()
    mlp = MLP(config_ld).build().save()
    print("Done!")
