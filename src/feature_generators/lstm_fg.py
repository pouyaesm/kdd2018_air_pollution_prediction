import pandas as pd
from src import util
from src.preprocess import reform
from src.preprocess import times
import settings
import const
import random
import time

class LSTMFG:

    def __init__(self, cfg, time_steps):
        """
        :param cfg:
        :param time_steps: number of hour values of x (input) per sample
        """
        self.config = cfg
        self.data = pd.DataFrame()  # time series data per station
        self.stations = pd.DataFrame()  # stations of time series
        self.features = pd.DataFrame()  # extracted features
        self.time_steps = time_steps
        self._train = pd.DataFrame()
        self._test = pd.DataFrame()
        self._station_count = 0
        self._valid_stations = pd.DataFrame()
        self._features_path = self.config[const.FEATURE_DIR] + \
                              self.config[const.FEATURE] + \
                              str(self.time_steps) + '_lstm.csv'

    def generate(self):
        # load_model data
        stations = pd.read_csv(self.config[const.STATIONS], sep=";", low_memory=False)
        ts = pd.read_csv(self.config[const.OBSERVED], sep=";", low_memory=False)
        data = reform.group_by_station(ts=ts, stations=stations)
        stations = stations.to_dict(orient='index')
        features = list()

        start_time = time.time()

        for _, s_info in stations.items():
            if s_info[const.PREDICT] != 1:
                continue
            station_id = s_info[const.ID]
            s_data = data[station_id]
            s_time = pd.to_datetime(s_data[const.TIME], format=const.T_FORMAT, utc=True).tolist()
            first_x = self.time_steps - 1
            last_x = len(s_data) - 1 - 48
            sid = [station_id] * (len(s_time) - self.time_steps)
            s_value = s_data[self.config[const.POLLUTANT]].tolist()
            t, value = reform.split(time=s_time, value=s_value, step=self.time_steps)
            label = times.split(time=s_time, value=s_value, group_hours=1, step=48, region=(first_x + 1, last_x + 1))
            # values to be predicted
            feature_set = [[s] + [t] + v + l for s, t, v, l in zip(sid, t, value, label)]
            features.extend(feature_set)
        # set name for columns
        columns = [const.ID, const.TIME]
        columns.extend(['v' + str(i) for i in range(0, self.time_steps)])
        columns.extend(['l' + str(i) for i in range(0, 48)])
        self.features = pd.DataFrame(data=features, columns=columns)
        print(len(self.features.index), 'feature vectors generated in',
              time.time() - start_time, 'secs')
        return self

    def next(self, batch_size, time_steps):
        if len(self._train.index) == 0:
            self.load()
        sample = self._train.sample(n=batch_size)
        values = sample.values
        x = util.row_to_matrix(values[:, 2:self.time_steps + 2], split_count=time_steps)
        # y = values[:, self.input_hours + 2:]
        # use reversed of output to make a closer connection
        # between last values of input and first values of output
        y = util.reverse(values[:, self.time_steps + 2:], axis=1)
        return x, y

    def test(self, time_steps):
        if len(self._test.index) == 0:
            self.load()
        x = util.row_to_matrix(self._test.values[:, 2:self.time_steps + 2], split_count=time_steps)
        # y = self._test.values[:, self.input_hours + 2:]
        # use reversed of output to make a closer connection
        # between last values of input and first values of output
        y = util.reverse(self._test.values[:, self.time_steps + 2:], axis=1)
        return x, y

    def load(self):
        features = pd.read_csv(self._features_path, sep=";", low_memory=False)
        self._train = times.select(df=features, time_key=const.TIME, from_time='00-01-01 00', to_time='18-03-29 00')
        # valid = times.select(df=ts, time_key=const.TIME, from_time='17-12-31 23', to_time='17-12-31 23')
        self._test = times.select(df=features, time_key=const.TIME, from_time='18-03-31 00', to_time='18-04-30 23')
        return self

    def dropna(self):
        self.features = self.features.dropna(axis=0)  # drop rows containing nan values
        return self

    def sample(self, count):
        if count > 0:
            self.features = self.features.sample(n=count)
        return self

    def save_features(self):
        """
            Save the extracted features to file
        :return:
        """
        util.write(self.features, address=self._features_path)
        print(len(self.features.index), 'feature vectors are written to file')

    def save_test(self, predicted_values):
        augmented_test = util.add_columns(self._test, columns=predicted_values, name_prefix='f')
        test_path = self.config[const.FEATURE_DIR] + self.config[const.FEATURE] + \
                    str(self.time_steps) + '_lstm_tests.csv'
        util.write(augmented_test, address=test_path)
        print(len(augmented_test.index), 'predicted tests are written to file')


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]
    cases = {
        'BJ': {
            'PM2.5': 200000,
            'PM10': -1,
            'O3': 200000,
            },
        'LD': {
            'PM2.5': -1,
            'PM10': -1,
            }
        }
    for city in cases:
        for pollutant, sample_count in cases[city].items():
            print(city, pollutant, "started..")
            cfg = {
                const.OBSERVED: config[getattr(const, city + '_OBSERVED')],
                const.STATIONS: config[getattr(const, city + '_STATIONS')],
                const.FEATURE_DIR: config[const.FEATURE_DIR],
                const.FEATURE: getattr(const, city + '_' + pollutant.replace('.', '') + '_'),
                const.POLLUTANT: pollutant
            }
            fg = LSTMFG(cfg, time_steps=48)
            fg.generate().dropna().sample(sample_count).save_features()
            print(city, pollutant, "done!")
