import const
import settings
import pandas as pd
import numpy as np
import random
from src.preprocess import reform, times
from src.feature_generators.lstm_fg import LSTMFG
from src import util
import time
import math


class HybridFG(LSTMFG):

    def __init__(self, cfg):
        """
        :param cfg:
        :param time_steps: number of values to be considered as input (for hour_x = 1) step_x = hour count
        """
        super(HybridFG, self).__init__(cfg, -1)

        # Basic parameters
        # Long features are more coarse grained and long range
        self.meo_steps = cfg[const.MEO_STEPS]  # time steps backward
        self.meo_group = cfg[const.MEO_GROUP]  # hours of each step
        self.meo_long_steps = cfg[const.MEO_LONG_STEPS]
        self.meo_long_group = 24
        self.future_steps = cfg[const.FUTURE_STEPS]
        self.future_group = cfg[const.FUTURE_GROUP]
        self.air_steps = cfg[const.AIR_STEPS]
        self.air_group = cfg[const.AIR_GROUP]
        self.air_long_steps = cfg[const.AIR_LONG_STEPS]
        self.air_long_group = 24
        self.time_is_one_hot = True

        self.meo_keys = [const.TEMP, const.HUM, const.WSPD]  # [const.TEMP, const.HUM, const.WSPD]
        self.future_keys = [const.TEMP, const.HUM, const.WSPD]
        # self.air_keys = [const.PM25]
        # self.air_keys = [const.PM25, const.PM10]  # [const.PM25, const.PM10, const.O3]
        if cfg[const.CITY] == const.BJ:
            self.air_keys = [const.PM25, const.PM10, const.O3]
        elif cfg[const.CITY] == const.LD:
            self.air_keys = [const.PM25, const.PM10]  # no O3 for london

        self.path_indicator = '%s_%s_%s_%s_%s_%s_%s_%s_%s_%s' % (self.meo_steps, self.meo_group,
                                                            self.meo_long_steps, self.meo_long_group,
                                                            self.future_steps, self.future_group,
                                                            self.air_steps, self.air_group,
                                                            self.air_long_steps, self.air_long_group)
        self._features_path = self.config.get(const.FEATURE_DIR, "") + \
                             self.config.get(const.FEATURE, "") + self.path_indicator + '_hybrid_'
        self._test_path = self._features_path + 'tests.csv'

        # number of file chunks to put features into
        self.chunk_count = self.config.get(const.CHUNK_COUNT, 1)

        self.train_from = self.config.get(const.TRAIN_FROM, '00-01-01 00')
        self.train_to = self.config.get(const.TRAIN_TO, '00-01-01 00')
        self.valid_from = self.config.get(const.VALID_FROM, '00-01-01 00')
        self.valid_to = self.config.get(const.VALID_TO, '00-01-01 00')
        self.test_from = self.config.get(const.TEST_FROM, '00-01-01 00')
        self.test_to = self.config.get(const.TEST_TO, '00-01-01 00')

        # train / test / validation data holders
        self._exploded = {const.TRAIN: dict(), const.VALID: dict(), const.TEST: dict()}

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
        station_count = self._stations[const.PREDICT].sum()
        processed_stations = 0
        next_chunk = 1
        total_data_points = 0
        for s_index, s_info in stations.items():
            if s_info[const.PREDICT] != 1:
                continue
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
            processed_stations += 1
            # save current chunk and go to next
            if save and (s_index >= chunk_index[next_chunk] or processed_stations == station_count):
                # set and save the chunk of features
                self.features = pd.DataFrame(data=features, columns=self.get_all_columns())
                before_drop = len(self.features)
                self.dropna()
                after_drop = len(self.features)
                print(' %d feature vectors dropped having NaN' % (before_drop - after_drop))
                self.save_features(chunk_id=next_chunk)
                total_data_points += len(self.features.index)
                # go to next chunk
                features = list()
                self.features = pd.DataFrame()
                next_chunk += 1

        if not save:
            self.features = pd.DataFrame(data=features, columns=self.get_all_columns())
            total_data_points = len(self.features)

        print(total_data_points, 'feature vectors generated in', time.time() - start_time, 'secs')
        return self

    def generate_per_station(self, station_id, s_data, s_time, first_x):
        # time of each data point (row)
        t = s_time[first_x:]  # first data point is started at 'first_x'

        region = (first_x, -1)  # region of values to be extracted as features

        #  Each data point (row): (measure1, measure2, ..) @ t0, (..) @ t1, .., (..) @ tN
        def reshape(list, row_size):
            return np.moveaxis(np.array(list), source=0, destination=2) \
                .reshape((-1, row_size)).tolist()

        # weather time series of last 'meo_steps' every 'meo_group' hours
        meo_all = list()
        for meo_key in self.meo_keys:
            ts = times.split(time=s_time, value=s_data[meo_key].tolist(),
                             group_hours=self.meo_group, step=-self.meo_steps, region=region)
            meo_all.append(ts)
        meo = reshape(meo_all, row_size=self.meo_steps * len(meo_all))

        # long range weather time series
        meo_long_all = list()
        for meo_key in self.meo_keys:
            ts = times.split(time=s_time, value=s_data[meo_key].tolist(),
                             group_hours=self.meo_long_group, step=-self.meo_long_steps, region=region)
            meo_long_all.append(ts)
        meo_long = reshape(meo_long_all, row_size=self.meo_long_steps * len(meo_long_all))

        # future weather time series of next 'future_steps' every 'future_group' hours
        future_all = list()
        for future_key in self.future_keys:
            ts = times.split(time=s_time, value=s_data[future_key].tolist(),
                             group_hours=self.future_group, step=self.future_steps, region=region,
                             whole_group=True)
            future_all.append(ts)
        future = reshape(future_all, row_size=self.future_steps * len(future_all))

        # air quality time series of last 'air_steps' every 'air_group' hours
        air_all = list()
        for air_key in self.air_keys:
            ts = times.split(time=s_time, value=s_data[air_key].tolist(),
                               group_hours=self.air_group, step=-self.air_steps, region=region)
            air_all.append(ts)
        air = reshape(air_all, row_size=self.air_steps * len(air_all))

        # long range air quality time series
        air_long_all = list()
        for air_key in self.air_keys:
            ts = times.split(time=s_time, value=s_data[air_key].tolist(),
                             group_hours=self.air_long_group, step=-self.air_long_steps, region=region)
            air_long_all.append(ts)
        air_long = reshape(air_long_all, row_size=self.air_long_steps * len(air_long_all))

        # next 48 pollutant values to be predicted
        pollutant = self.config[const.POLLUTANT]
        label = times.split(time=s_time, value=s_data[pollutant].tolist(),
                            group_hours=1, step=48, region=(first_x + 1, -1))
        # station id per row
        sid = [station_id] * (len(s_time) - first_x)

        # aggregate all features per row
        feature_set = [[s] + [t] + m + ml + f + al + a + l for s, t, m, ml, f, al, a, l in
                       zip(sid, t, meo, meo_long, future, air_long, air, label)]

        return feature_set

    def next(self, batch_size, progress=0, rotate=1):
        """
            Next batch for training
        :param batch_size:
        :param progress: progress ratio to be used to move to next feature file chunks
        :param rotate: number of times to rotate over all chunks until progress = 1
        :return: tuple (context, meo_ts, future_ts, air_ts, label)
        :rtype: (list, list, list, list, list)
        """
        chunk_id = 1 + math.floor(rotate * progress * self.chunk_count) % self.chunk_count
        if self._current_chunk != chunk_id:
            self.load(chunk_id=chunk_id)
            self._current_chunk = chunk_id
        index = const.TRAIN
        exploded = self._exploded[index]
        sample_idx = np.random.randint(len(exploded['c']), size=batch_size)
        context = exploded['c'][sample_idx, :]
        meo = exploded['m'][sample_idx, :]
        meo_long = exploded['ml'][sample_idx, :]
        future = exploded['f'][sample_idx, :]
        air = exploded['a'][sample_idx, :]
        air_long = exploded['al'][sample_idx, :]
        label = exploded['l'][sample_idx, :]
        return {
            'c': context,
            'm': meo,
            'ml': meo_long,
            'f': future,
            'a': air,
            'al': air_long,
            'l': label
        }

    def holdout(self, key=const.TEST):
        """
            Return test data
        :param key: key for TEST or VALID data
        :return: tuple (context, meo_ts, future_ts, air_ts, label)
        :rtype: (list, list, list, list, list)
        """
        if len(self._exploded[key]) == 0:
            self.load_holdout()
        return self._exploded[key]

    def load(self, chunk_id=1):
        """
            Load a chunk of training data, separated into different inputs
        :param chunk_id:
        :return:
        """
        features = pd.read_csv(self._features_path + str(chunk_id) + '.csv', sep=";", low_memory=False)
        train_features = times.select(df=features, time_key=const.TIME,
                                   from_time=self.train_from, to_time=self.train_to)
        self._exploded[const.TRAIN] = self.explode(train_features)
        print('Feature chunk {c} is prepared.'.format(c=chunk_id))
        return self

    def load_holdout(self):
        print(' Load validation set from %s to %s' % (self.valid_from, self.valid_to))
        print(' Load test set from %s to %s' % (self.test_from, self.test_to))
        for chunk_id in range(1, self.chunk_count + 1):
            input_features = pd.read_csv(self._features_path + str(chunk_id) + '.csv', sep=";", low_memory=False)
            # extract test and validation data
            features = dict()
            features[const.VALID] = times.select(df=input_features, time_key=const.TIME,
                                                 from_time=self.valid_from, to_time=self.valid_to)
            features[const.TEST] = times.select(df=input_features, time_key=const.TIME,
                                                from_time=self.test_from, to_time=self.test_to)
            # add feature to global test data
            if len(self._test.index) == 0:
                self._test = features[const.TEST]
            else:
                self._test = self._test.append(other=features[const.TEST], ignore_index=True)

            # explode features into parts (context, weather time series, etc.)
            for key in features:
                exploded = self.explode(features[key])
                if len(exploded) == 0:
                    continue
                for part, value in exploded.items():
                    self._exploded[key][part] = value if part not in self._exploded[key] \
                        else np.concatenate((self._exploded[key][part], value), axis=0)
        print(' Hold-out feature is prepared (valid: %d, test: %d).' % (
            len(self._exploded[const.VALID]['c']), len(self._exploded[const.TEST]['c'])))
        return self

    def explode(self, features: pd.DataFrame):
        """
            Explode features to context, time series, and label
        :param features:
        :return: exploded feature parts
        :rtype: (dict)
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
        meo = new_features[self.get_meo_columns()].as_matrix()
        meo = util.row_to_matrix(meo, split_count=self.meo_steps)
        # weather long range time series
        meo_long = new_features[self.get_meo_columns(is_long=True)].as_matrix()
        meo_long = util.row_to_matrix(meo_long, split_count=self.meo_long_steps)
        # forecast weather time series
        future = new_features[self.get_future_columns()].as_matrix()
        future = util.row_to_matrix(future, split_count=self.future_steps)
        # air quality time series
        air = new_features[self.get_air_columns()].as_matrix()
        air = util.row_to_matrix(air, split_count=self.air_steps)
        # air quality long range time series
        air_long = new_features[self.get_air_columns(is_long=True)].as_matrix()
        air_long = util.row_to_matrix(air_long, split_count=self.air_long_steps)
        # label time series
        label = new_features[self.get_label_columns()].as_matrix()

        return {
            'c': context,
            'm': meo,
            'ml': meo_long,
            'f': future,
            'a': air,
            'al': air_long,
            'l': label
        }

    def shuffle(self):
        shuffle_count = self.chunk_count - 1
        shuffle_counter = 0
        for iteration in range(0, shuffle_count):
            chunk1_id = 2 + iteration % (self.chunk_count - 1)
            chunk2_id = random.randint(1, chunk1_id - 1)
            print(' Shuffling files {id1} <-> {id2} ..'.format(id1=chunk1_id, id2=chunk2_id))
            chunk1_path = self._features_path + str(chunk1_id) + '.csv'
            chunk2_path = self._features_path + str(chunk2_id) + '.csv'
            chunk1 = pd.read_csv(chunk1_path, sep=";", low_memory=False)
            chunk2 = pd.read_csv(chunk2_path, sep=";", low_memory=False)
            # merge two chunks and shuffle the merged rows
            chunk1 = chunk1.append(other=chunk2, ignore_index=True).sample(
                frac=1).reset_index(drop=True)
            # save shuffled chunks
            border = int(len(chunk1.index) / 2)
            util.write(chunk1.ix[0:border, :], address=chunk1_path)
            util.write(chunk1.ix[border:, :], address=chunk2_path)
            shuffle_counter += 1
            print(' Feature files {id1} <-> {id2} shuffled ({c} of {t})'.format(
                id1=chunk1_id, id2=chunk2_id, c=shuffle_counter, t=shuffle_count))
        return self

    def dropna(self):
        self.features = self.features.dropna(axis=0)  # drop rows containing nan values
        return self

    def get_context_count(self):
        if self.time_is_one_hot:
            return 2 + 12 + 7  # long, lat, hour // 2, day_of_week
        else:
            return 2 + 1 + 1  # long, lat, hour, day_of_week

    def get_measured_columns(self):
        # order of these lines is important
        # order of output of generate_per_station is connected to this
        columns = self.get_meo_columns()
        columns.extend(self.get_meo_columns(is_long=True))
        columns.extend(self.get_future_columns())
        columns.extend(self.get_air_columns(is_long=True))
        columns.extend(self.get_air_columns())
        columns.extend(self.get_label_columns())
        return columns

    def get_all_columns(self):
        # set name for columns
        columns = [const.ID, const.TIME]
        columns.extend(self.get_measured_columns())
        return columns

    def get_label_columns(self):
        return [self.config[const.POLLUTANT] + '__' + str(i) for i in range(1, 49)]

    def get_meo_columns(self, is_long=False):
        columns = []
        steps = self.meo_long_steps if is_long else self.meo_steps
        template = '{k}_l_{i}' if is_long else '{k}_{i}'
        for i in range(1, steps + 1):
            for meo_key in self.meo_keys:
                columns.extend([template.format(k=meo_key, i=i)])
        return columns

    def get_future_columns(self):
        columns = []
        for i in range(1, self.future_steps + 1):
            for future_key in self.future_keys:
                columns.extend(['{k}_f_{i}'.format(k=future_key, i=i)])
        return columns

    def get_air_columns(self, is_long=False):
        columns = []
        steps = self.air_long_steps if is_long else self.air_steps
        template = '{k}_l_{i}' if is_long else '{k}_{i}'
        for i in range(1, steps + 1):
            for air_key in self.air_keys:
                columns.extend([template.format(k=air_key, i=i)])
        return columns

    def get_features(self):
        """
        :return:
        :rtype: pandas.DataFrame
        """
        return self.features

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

    @staticmethod
    def get_size_config(city, key='default'):
        if key == 'default':
            return {
                const.CHUNK_COUNT: 10 if city == const.BJ else 4,
                # weather of past 36 hours
                const.MEO_STEPS: 12,
                const.MEO_GROUP: 3,
                # weather of past 7for days
                const.MEO_LONG_STEPS: 7,  # in days
                # weather of next 48 hours
                const.FUTURE_STEPS: 8,
                const.FUTURE_GROUP: 6,
                # air quality of past 12 hours
                const.AIR_STEPS: 12,
                const.AIR_GROUP: 1,
                # air quality of past 7 days
                const.AIR_LONG_STEPS: 7,  # in days
            }
        return {}


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]
    cases = {
        'BJ': {
            # 'PM2.5': True,
            # 'PM10': True,
            # 'O3': True,
            },
        'LD': {
            'PM2.5': True,
            'PM10': True,
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
            }
            cfg.update(HybridFG.get_size_config(city=city))  # configuration of feature sizes
            fg = HybridFG(cfg=cfg)
            fg.generate()
            fg.shuffle()  # shuffle generated chunks for data uniformity in learning phase
            print(city, pollutant, "done!")
