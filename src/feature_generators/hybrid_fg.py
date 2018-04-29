import const
import settings
import pandas as pd
import numpy as np
from src.preprocess import reform, times
from src.feature_generators.lstm_fg import LSTMFG
from src import util
import time
import math


class HybridFG(LSTMFG):

    def __init__(self, time_steps, cfg):
        """
        :param cfg:
        :param time_steps: number of values to be considered as input (for hour_x = 1) step_x = hour count
        """
        super(HybridFG, self).__init__(cfg, time_steps)

        # Basic parameters
        self.meo_steps = 24
        self.meo_group = 3
        self.air_steps = 12
        self.air_group = 1
        self.time_is_one_hot = True
        self.meo_keys = [const.TEMP, const.HUM, const.WSPD]  # [const.TEMP, const.HUM, const.WSPD]
        # self.air_keys = [const.PM25]
        # self.air_keys = [const.PM25, const.PM10]  # [const.PM25, const.PM10, const.O3]
        self.air_keys = [const.PM25, const.PM10, const.O3]

        features_base_path = self.config[const.FEATURE_DIR] + self.config[const.FEATURE] + str(self.time_steps)
        self._features_path = features_base_path + '_hybrid_'

        self._test_path = features_base_path + '_hybrid_tests.csv'

        # number of file chunks to put features into
        self.chunk_count = self.config.get(const.CHUNK_COUNT, 1)

        self.train_from = '00-01-01 00'
        self.train_to = '18-03-13 23'
        self.valid_from = '18-03-15 23'
        self.valid_to = '18-03-29 23'

        # train / test / validation data holders
        self._context = [np.empty((0, 0))] * 3
        self._meo = [np.empty((0, 0))] * 3
        self._air = [np.empty((0, 0))] * 3
        self._label = [np.empty((0, 0))] * 3

        # station data for context features like station location
        self._stations = pd.DataFrame()

        self._current_chunk = -1  # current feature file chunk

    def generate(self, ts=None, stations=None, verbose=True, save=True):
        """
            Create a basic feature set from pollutant time series, per hour
            x: (time, longitude, latitude, pollutant values of t:t+n)
            y: (pollutant values of t+n:t+n+m)
        :return:
        """
        # load_model data
        if ts is None:
            ts = pd.read_csv(self.config[const.OBSERVED], sep=";", low_memory=False)
        if stations is None:
            self._stations = pd.read_csv(self.config[const.STATIONS], sep=";", low_memory=False)
        else:
            self._stations = stations

        self.data = reform.group_by_station(ts=ts, stations=self._stations)

        features = list()
        start_time = time.time()
        stations = self._stations.to_dict(orient='index')
        chunk_index = np.linspace(start=0, stop=len(stations) - 1, num=self.chunk_count + 1)
        next_chunk = 1
        total_data_points = 0
        for s_index, s_info in stations.items():
            if s_info[const.PREDICT] != 1: continue
            station_id = s_info[const.ID]
            if verbose:
                print(' Features of {sid} ({index} of {len})..'.
                      format(sid=station_id, index=s_index + 1, len=len(stations)))
            s_data = self.data[station_id]
            s_time = pd.to_datetime(s_data[const.TIME], format=const.T_FORMAT).tolist()
            first_x = self.air_group * self.air_steps - 1
            station_features = self.generate_per_station(station_id, s_data, s_time, first_x)
            # aggregate all features per row
            features.extend(station_features)
            # save current chunk and go to next
            if save and s_index >= chunk_index[next_chunk]:
                # set and save the chunk of features
                self.features = pd.DataFrame(data=features, columns=self.get_all_columns())
                self.dropna().save_features(chunk_id=next_chunk)
                total_data_points += len(self.features.index)
                # go to next chunk
                features = list()
                self.features = pd.DataFrame()
                next_chunk += 1

        if not save:
            self.features = pd.DataFrame(data=features, columns=self.get_all_columns())

        print(total_data_points, 'feature vectors generated in',
              time.time() - start_time, 'secs')
        return self

    def generate_per_station(self, station_id, s_data, s_time, first_x):
        # time of each data point (row)
        t = s_time[first_x:]  # first data point is started at 'first_x'

        region = (first_x, -1)  # region of values to be extracted as features
        # weather time series of last 'meo_steps' every 'meo_group' hours
        meo_all = list()
        for meo_key in self.meo_keys:
            ts = times.split(time=s_time, value=s_data[meo_key].tolist(),
                             group_hours=self.meo_group, step=-self.meo_steps, region=region)
            meo_all.append(ts)
        # each data point (row): (temp, hum, wspd) @ t0, (..) @ t1, .., (..) @ tN
        # order is important for column names
        meo_ts = np.moveaxis(np.array(meo_all), source=0, destination=2) \
            .reshape((-1, self.meo_steps * len(meo_all))).tolist()

        # air quality time series of last 'air_steps' every 'air_group' hours
        air_all = list()
        for air_key in self.air_keys:
            ts = times.split(time=s_time, value=s_data[air_key].tolist(),
                               group_hours=self.air_group, step=-self.air_steps, region=region)
            air_all.append(ts)
        # each data point (row): (pm25, pm10, o3) @ t0, (..) @ t1, .., (..) @ tN
        air_ts = np.moveaxis(np.array(air_all), source=0, destination=2) \
            .reshape((-1, self.air_steps * len(air_all))).tolist()

        # next 48 pollutant values to be predicted
        pollutant = self.config[const.POLLUTANT]
        label = times.split(time=s_time, value=s_data[pollutant].tolist(),
                            group_hours=1, step=48, region=(first_x + 1, -1))
        # station id per row
        sid = [station_id] * (len(s_time) - first_x)

        # aggregate all features per row
        feature_set = [[s] + [t] + m + a + l for s, t, m, a, l in zip(sid, t, meo_ts, air_ts, label)]

        return feature_set

    def next(self, batch_size, progress=0, rotate=1):
        """
            Next batch for training
        :param batch_size:
        :param progress: progress ratio to be used to move to next feature file chunks
        :param rotate: number of times to rotate over all chunks until progress = 1
        :return: tuple (context, meo_ts, air_ts, label)
        :rtype: (list, list, list, list)
        """
        chunk_id = 1 + math.floor(rotate * progress * self.chunk_count) % self.chunk_count
        if self._current_chunk != chunk_id:
            self.load(chunk_id=chunk_id)
            self._current_chunk = chunk_id
        index = const.TRAIN
        sample_idx = np.random.randint(len(self._context[index]), size=batch_size)
        sample_idx[0] = 0
        context = self._context[index][sample_idx, :]
        meo_ts = self._meo[index][sample_idx, :]
        air_ts = self._air[index][sample_idx, :]
        label = self._label[index][sample_idx, :]
        return context, meo_ts, air_ts, label

    def holdout(self, key=const.TEST):
        """
            Return test data
        :param key: key for TEST or VALID data
        :return: tuple (context, meo_ts, air_ts, label)
        :rtype: (list, list, list, list)
        """
        if len(self._context[key]) == 0:
            self.load_valid_test()
        return self._context[key], self._meo[key], self._air[key], self._label[key]

    def load(self, chunk_id=1):
        """
            Load a chunk of training data, separated into different inputs
        :param chunk_id:
        :return:
        """
        features = pd.read_csv(self._features_path + str(chunk_id) + '.csv', sep=";", low_memory=False)
        train_features = times.select(df=features, time_key=const.TIME,
                                   from_time=self.train_from, to_time=self.train_to)
        context, meo_ts, air_ts, label = self.explode(train_features)
        index = const.TRAIN
        self._context[index] = context
        self._meo[index] = meo_ts
        self._air[index] = air_ts
        self._label[index] = label
        print('Feature chunk {c} is prepared.'.format(c=chunk_id))
        return self

    def load_valid_test(self):
        test_from = self.config[const.TEST_FROM]
        test_to = self.config[const.TEST_TO]
        for chunk_id in range(1, self.chunk_count + 1):
            input_features = pd.read_csv(self._features_path + str(chunk_id) + '.csv', sep=";", low_memory=False)
            # extract test and validation data
            features = dict()
            features[const.TEST] = times.select(df=input_features, time_key=const.TIME,
                                                from_time=test_from, to_time=test_to)
            features[const.VALID] = times.select(df=input_features, time_key=const.TIME,
                                          from_time=self.valid_from, to_time=self.valid_to)
            # add feature to global test data
            if len(self._test.index) == 0:
                self._test = features[const.TEST]
            else:
                self._test = self._test.append(other=features[const.TEST], ignore_index=True)

            # explode features into parts (context, weather time series, etc.)
            for key in features:
                context, meo_ts, air_ts, label = self.explode(features[key])
                if len(self._context[key]) == 0:
                    self._context[key] = context
                    self._meo[key] = meo_ts
                    self._air[key] = air_ts
                    self._label[key] = label
                else:
                    self._context[key] = np.concatenate((self._context[key], context), axis=0)
                    self._meo[key] = np.concatenate((self._meo[key], meo_ts), axis=0)
                    self._air[key] = np.concatenate((self._air[key], air_ts), axis=0)
                    self._label[key] = np.concatenate((self._label[key], label), axis=0)
        print(' Hold-out feature is prepared.')
        return self

    def explode(self, features: pd.DataFrame):
        """
            Explode features to context, time series, and label
        :param features:
        :return: tuple (context, meo_ts, air_ts, label)
        :rtype: (list, list, list, list)
        """
        if len(self._stations.index) == 0:
            self._stations = pd.read_csv(self.config[const.STATIONS], sep=";", low_memory=False)
        new_features = features.merge(right=self._stations, how='left', on=const.ID)

        # normalize all measured feature values
        # extract basic names from column names by removing indices, e.g. PM10_2 -> PM10
        # names = [column.split('_')[0] for column in self.get_measured_columns()]
        # city = self.config[const.CITY]
        # for index, column in enumerate(self.get_measured_columns()):
        #     new_features[column] = FG.normalize(new_features[column], name=names[index], city=city)

        position = new_features[[const.LONG, const.LAT]].as_matrix().tolist()
        features_time = pd.to_datetime(new_features[const.TIME], format=const.T_FORMAT)

        if self.time_is_one_hot:
            # Extract even hours as a one-hot vector
            hour_values = ["%02d" % h for h in range(0, 12)]  # only consider even hours to have 12 bits
            hours = pd.to_datetime(features_time.dt.hour // 2, format='%H')
            hours_oh = times.one_hot(times=hours, columns=hour_values, time_format='%H').values.tolist()
            # Extract days of week as one-hot vector
            dow_values = ["%d" % dow for dow in range(0, 7)]  # days of week
            # strftime.sunday = 0
            dows_oh = times.one_hot(times=features_time, columns=dow_values, time_format='%w').values.tolist()
            # long, lat, one-hot hour, one hot day of week
            context = np.array([p + h + dow for p, h, dow in zip(position, hours_oh, dows_oh)], dtype=np.float32)
        else:
            hours = features_time.dt.hour
            dows = (features_time.dt.dayofweek + 1) % 7
            # long, lat, one-hot hour, one hot day of week
            context = np.array([p + [h] + [dow] for p, h, dow in zip(position, hours, dows)], dtype=np.float32)

        # weather time series
        meo_ts = new_features[self.get_meo_columns()].as_matrix()
        meo_ts = util.row_to_matrix(meo_ts, split_count=self.meo_steps)
        # air quality time series
        air_ts = new_features[self.get_air_columns()].as_matrix()
        air_ts = util.row_to_matrix(air_ts, split_count=self.air_steps)
        # label time series
        label = new_features[self.get_label_columns()].as_matrix()

        return context, meo_ts, air_ts, label

    def dropna(self):
        self.features = self.features.dropna(axis=0)  # drop rows containing nan values
        return self

    def get_context_count(self):
        if self.time_is_one_hot:
            return 2 + 12 + 7  # long, lat, hour // 2, day_of_week
        else:
            return 2 + 1 + 1  # long, lat, hour, day_of_week

    def get_measured_columns(self):
        columns = self.get_meo_columns()
        columns.extend(self.get_air_columns())
        columns.extend(self.get_label_columns())
        return columns

    def get_all_columns(self):
        # set name for columns
        columns = [const.ID, const.TIME]
        columns.extend(self.get_meo_columns())
        columns.extend(self.get_air_columns())
        columns.extend(self.get_label_columns())
        return columns

    def get_label_columns(self):
        return [self.config[const.POLLUTANT] + '__' + str(i) for i in range(1, 49)]

    def get_meo_columns(self):
        columns = []
        for i in range(1, self.meo_steps + 1):
            for meo_key in self.meo_keys:
                columns.extend(['{k}_{i}'.format(k=meo_key, i=i)])
        return columns

    def get_features(self):
        return self.features

    def get_air_columns(self):
        columns = []
        for i in range(1, self.air_steps + 1):
            for air_key in self.air_keys:
                columns.extend(['{k}_{i}'.format(k=air_key, i=i)])
        return columns

    def sample(self, count):
        if 0 < count < len(self.features.index):
            self.features = self.features.sample(n=count)
        return self

    def save_features(self, chunk_id=1):
        """
            Save the extracted features to file
        :return:
        """
        util.write(self.features, address=self._features_path + str(chunk_id) + '.csv')
        print(len(self.features.index), 'feature vectors are written to file')

    def save_test(self, predicted_values):
        augmented_test = util.add_columns(self._test, columns=predicted_values, name_prefix='f')
        util.write(augmented_test, address=self._test_path)
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
                const.CITY: city,
                const.OBSERVED: config[getattr(const, city + '_OBSERVED')],
                const.STATIONS: config[getattr(const, city + '_STATIONS')],
                const.FEATURE_DIR: config[const.FEATURE_DIR],
                const.FEATURE: getattr(const, city + '_' + pollutant.replace('.', '') + '_'),
                const.POLLUTANT: pollutant,
                const.CHUNK_COUNT: 8,
            }
            fg = HybridFG(cfg=cfg, time_steps=12)
            fg.generate()
            print(city, pollutant, "done!")
