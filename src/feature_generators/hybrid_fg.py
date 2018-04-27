import const
import settings
import pandas as pd
import numpy as np
from src.preprocess import reform, times
from src.feature_generators.lstm_fg import LSTMFG
from src import util
import time


class HybridFG(LSTMFG):

    def __init__(self, cfg, time_steps):
        """
        :param cfg:
        :param time_steps: number of values to be considered as input (for hour_x = 1) step_x = hour count
        :param group_hours: number of hour values of x (input) per step
        """
        super(HybridFG, self).__init__(cfg, time_steps)
        self._features_path = self.config[const.FEATURE_DIR] + \
                              self.config[const.FEATURE] + \
                              str(self.time_steps) + '_hybrid.csv'
        # Basic parameters
        self.meo_steps = 3
        self.meo_group = 1
        self.air_steps = 3
        self.air_group = 1
    def generate(self):
        """
            Create a basic feature set from pollutant time series, per hour
            x: (time, longitude, latitude, pollutant values of t:t+n)
            y: (pollutant values of t+n:t+n+m)
        :return:
        """

        # load_model data
        self.stations = pd.read_csv(self.config[const.STATIONS], sep=";", low_memory=False)
        ts = pd.read_csv(self.config[const.OBSERVED], sep=";", low_memory=False)
        self.data = reform.group_by_station(ts=ts, stations=self.stations)

        pollutant = self.config[const.POLLUTANT]
        # columns = ['forecast', 'actual', 'station', 'pollutant']
        features = list()
        # labels = list()
        start_time = time.time()
        stations = self.stations.to_dict(orient='index')
        for s_index, s_info in stations.items():
            if s_info[const.PREDICT] != 1: continue
            station_id = s_info[const.ID]
            print(' Features of {sid} ({index} of {len})..'.
                  format(sid=station_id, index=s_index, len=len(stations)))
            s_data = self.data[station_id]
            s_time = pd.to_datetime(s_data[const.TIME], format=const.T_FORMAT, utc=True).tolist()
            first_x = self.air_group * self.air_steps - 1
            last_x = len(s_time) - first_x - 48 - 1
            # time of each data point (row)
            t = s_time[first_x:last_x + 1]  # first data point is started at 'first_x'

            region = (first_x, last_x)  # region of values to be extracted as features
            # weather time series of last 'meo_steps' every 'meo_group' hours
            temp = times.split(time=s_time, value=s_data[const.TEMP].tolist(),
                               group_hours=self.meo_group, step=-self.meo_steps, region=region)
            hum = times.split(time=s_time, value=s_data[const.HUM].tolist(),
                              group_hours=self.meo_group, step=-self.meo_steps, region=region)
            wspd = times.split(time=s_time, value=s_data[const.WSPD].tolist(),
                               group_hours=self.meo_group, step=-self.meo_steps, region=region)
            # each data point (row): (temp, hum, wspd) @ t0, (..) @ t1, .., (..) @ tN
            # order is important for column names
            meo_ts = np.moveaxis(np.array([temp, hum, wspd]), source=0, destination=2) \
                .reshape((-1, self.meo_steps * 3)).tolist()

            # air quality time series of last 'air_steps' every 'air_group' hours
            pm25 = times.split(time=s_time, value=s_data[const.PM25].tolist(),
                               group_hours=self.air_group, step=-self.air_steps, region=region)
            pm10 = times.split(time=s_time, value=s_data[const.PM10].tolist(),
                               group_hours=self.air_group, step=-self.air_steps, region=region)
            o3 = times.split(time=s_time, value=s_data[const.O3].tolist(),
                             group_hours=self.air_group, step=-self.air_steps, region=region)
            # each data point (row): (pm25, pm10, o3) @ t0, (..) @ t1, .., (..) @ tN
            air_ts = np.moveaxis(np.array([pm25, pm10, o3]), source=0, destination=2)\
                .reshape((-1, self.air_steps * 3)).tolist()

            # next 48 pollutant values to be predicted
            label = times.split(time=s_time, value=s_data[pollutant].tolist(),
                                group_hours=1, step=48, region=(first_x + 1, last_x + 1))
            # station id per row
            sid = [station_id] * (len(s_time) - first_x)

            # aggregate all features per row
            feature_set = [[s]+[t]+m+a+l for s, t, m, a, l in zip(sid, t, meo_ts, air_ts, label)]
            features.extend(feature_set)
        # set name for columns
        columns = [const.ID, const.TIME]
        for i in range(1, self.meo_steps + 1):
            columns.extend(['temp_' + str(i), 'hum_' + str(i), 'wspd_' + str(i)])
        for i in range(1, self.air_steps + 1):
            columns.extend(['pm25_' + str(i), 'pm10_' + str(i), 'o3_' + str(i)])
        columns.extend(['l_' + str(i) for i in range(1, 49)])
        self.features = pd.DataFrame(data=features, columns=columns)
        print(len(self.features.index), 'feature vectors generated in',
              time.time() - start_time, 'secs')
        return self

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
        if count > 0 and len(self.features.index) > 0:
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
            # 'PM10': -1,
            # 'O3': 200000,
            },
        'LD': {
            # 'PM2.5': -1,
            # 'PM10': -1,
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
            fg = HybridFG(cfg, time_steps=48)
            fg.generate().dropna().sample(sample_count).save_features()
            print(city, pollutant, "done!")
