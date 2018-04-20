import pandas as pd
from src import util
from src.preprocess import reform
from src.preprocess import times
import settings
import const
import random
import time

class LSTMFG:

    def __init__(self, cfg, input_hours):
        """
        :param cfg:
        :param input_hours: number of hour values of x (input) per sample
        """
        self.config = cfg
        self.data = pd.DataFrame()  # time series data per station
        self.stations = pd.DataFrame()  # stations of time series
        self.features = pd.DataFrame()  # extracted features
        self.input_hours = input_hours
        self._train = pd.DataFrame()
        self._test = pd.DataFrame()
        self._station_count = 0
        self._valid_stations = pd.DataFrame()

    def generate(self):
        # load data
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
            first_x_end = self.input_hours - 1
            # temperature of last 7 days every 3 hours
            # temp_h6_28 = times.split(time=s_time, value=s_data[const.TEMP].tolist(),
            #                          hours=6, step=28, skip=first_x_end)
            # wspd_h6_28 = times.split(time=s_time, value=s_data[const.WSPD].tolist(),
            #                          hours=6, step=28, skip=first_x_end)
            # dayofweek = [time.dayofweek +  for time in enumerate(t)]
            # location of station
            # loc = [[s_info[const.LONG], s_info[const.LAT]]] * (len(s_time) - self.input_hours)
            sid = [station_id] * (len(s_time) - self.input_hours)
            s_value = s_data[self.config[const.POLLUTANT]].tolist()
            t, value = reform.split(time=s_time, value=s_value, step=self.input_hours)
            label = times.split(time=s_time, value=s_value, hours=1, step=48, skip=first_x_end + 48)
            # temperature = times.split(time=s_time, value=s_data[const.TEMP], hours=1,
            #                      step=self.input_hours, skip=first_x_end)
            # pressure = times.split(time=s_time, value=s_data[const.PRES], hours=1,
            #                    step=self.input_hours, skip=first_x_end)
            # humidity = times.split(time=s_time, value=s_data[const.HUM], hours=1,
            #                    step=self.input_hours, skip=first_x_end)
            # wind_speed = times.split(time=s_time, value=s_data[const.WSPD], hours=1,
            #                        step=self.input_hours, skip=first_x_end)
            # wind_direction = times.split(time=s_time, value=s_data[const.WDIR], hours=1,
            #                          step=self.input_hours, skip=first_x_end)
            # values to be predicted
            feature_set = [[s] + [t] + v + l for s, t, v, l in zip(sid, t, value, label)]
            features.extend(feature_set)
        # set name for columns
        columns = [const.ID, const.TIME]
        columns.extend(['v' + str(i) for i in range(0, self.input_hours)])
        columns.extend(['l' + str(i) for i in range(0, 48)])
        self.features = pd.DataFrame(data=features, columns=columns)
        print(len(self.features.index), 'feature vectors generated in',
              time.time() - start_time, 'secs')
        return self

    def next(self, batch_size, time_steps):
        if len(self._train.index) == 0:
            self.load_for_next()
        sample = self._train.sample(n=batch_size)
        values = sample.values
        x = util.row_to_matrix(values[:, 2:self.input_hours + 2], row_split=time_steps)
        y = values[:, self.input_hours + 2:]
        return x, y

    def test(self, time_steps):
        x = util.row_to_matrix(self._train.values[:, 2:self.input_hours + 2], row_split=time_steps)
        y = self._train.values[:, self.input_hours + 2:]
        return x, y

    def load_for_next(self):
        features = pd.read_csv(self.config[const.FEATURES], sep=";", low_memory=False)
        self._train = times.select(df=features, time_key=const.TIME, from_time='00-01-01 00', to_time='17-12-31 23')
        # valid = times.select(df=ts, time_key=const.TIME, from_time='17-12-31 23', to_time='17-12-31 23')
        self._test = times.select(df=features, time_key=const.TIME, from_time='18-01-01 00', to_time='18-02-02 23')

    def dropna(self):
        self.features = self.features.dropna(axis=0)  # drop rows containing nan values
        return self

    def sample(self, count):
        self.features = self.features.sample(n=count)
        return self

    def save(self):
        """
            Save the extracted features to file
        :return:
        """
        util.write(self.features, address=self.config[const.FEATURES])
        print(len(self.features.index), ' feature vectors are written to file')


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]
    pollutant = 'PM2.5'
    features_bj = config[getattr(const, 'BJ_' + pollutant.replace('.', '') + '_')] + 'lstm_features.csv'
    features_ld = config[getattr(const, 'LD_' + pollutant.replace('.', '') + '_')] + 'lstm_features.csv'
    config_bj = {
        const.OBSERVED: config[const.BJ_OBSERVED],
        const.STATIONS: config[const.BJ_STATIONS],
        const.FEATURES: features_bj,
        const.POLLUTANT: pollutant
    }
    config_ld = {
        const.OBSERVED: config[const.LD_OBSERVED],
        const.STATIONS: config[const.LD_STATIONS],
        const.FEATURES: features_ld,
        const.POLLUTANT: pollutant
    }
    fg = LSTMFG(config_bj, input_hours=48)
    # fg = LSTMFG(config_ld, input_hours=48)
    fg.generate().dropna().save()
    print("Done!")
