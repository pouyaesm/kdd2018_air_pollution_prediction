import pandas as pd
from src import util
from src.preprocess import reform
from src.preprocess import times
import settings
import const
import random
import time

class LSTMFG:

    def __init__(self, cfg, pollutant, input_hours):
        """
        :param cfg:
        :param input_hours: number of hour values of x (input) per sample
        """
        self.config = cfg
        self.data = pd.DataFrame()  # time series data per station
        self.stations = pd.DataFrame()  # stations of time series
        self.features = pd.DataFrame()  # extracted features
        self.input_hours = input_hours
        self.pollutant = pollutant
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
            s_data = data[s_info[const.ID]]
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
            s_value = s_data[self.pollutant].tolist()
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
            feature_set = [v + l for v, l in zip(value, label)]
            features.extend(feature_set)
        # set name for columns
        columns = []  # [const.TIME, const.LONG, const.LAT]
        columns.extend(['v' + str(i) for i in range(0, self.input_hours)])
        columns.extend(['l' + str(i) for i in range(0, 48)])
        self.features = pd.DataFrame(data=features, columns=columns)
        print(len(self.features.index), 'feature vectors generated in',
              time.time() - start_time, 'secs')
        return self

    def next(self):
        if len(self._train.index) == 0:
            self.load_for_next()
        x = 1
        y = 1
        return x, y

    def load_for_next(self):
        self.stations = pd.read_csv(self.config[const.STATIONS], sep=";", low_memory=False)
        ts = pd.read_csv(self.config[const.OBSERVED], sep=";", low_memory=False)
        train = times.select(df=ts, time_key=const.TIME, from_time='00-01-01 00', to_time='17-12-31 23')
        # valid = times.select(df=ts, time_key=const.TIME, from_time='17-12-31 23', to_time='17-12-31 23')
        test = times.select(df=ts, time_key=const.TIME, from_time='18-01-01 00', to_time='18-02-02 23')
        self._train = reform.group_by_station(ts=train, stations=self.stations)
        self._test = reform.group_by_station(ts=test, stations=self.stations)
        self._valid_stations = self.stations.loc[self.stations[const.PREDICT] == 1, :]
        self._valid_stations.reset_index(inplace=True)

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
    fg = LSTMFG({
        const.OBSERVED: config[const.BJ_OBSERVED],
        const.STATIONS: config[const.BJ_STATIONS],
        const.FEATURES: config[const.BJ_PM10_] + 'lstm_features.csv',
    }, pollutant='PM10', input_hours=48)
    fg.generate().dropna().save()
    print("Done!")
