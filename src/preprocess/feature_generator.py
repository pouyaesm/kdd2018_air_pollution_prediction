import const, settings
from src.preprocess import reform, times
from src import util
import time
import pandas as pd
import numpy as np


class FeatureGenerator:

    def __init__(self, config, hour_x, hour_y):
        """
        :param config:
        :param hour_x: number of hour values of x (input) per sample
        :param hour_y: number of hour values of y (output) per sample
        """
        self.config = config
        self.data = pd.DataFrame()  # time series data per station
        self.stations = pd.DataFrame  # stations of time series
        self.features = pd.DataFrame()  # extracted features
        self.hour_x = hour_x
        self.hour_y = hour_y

    def load(self):
        # load data
        self.stations = pd.read_csv(self.config[const.STATIONS], sep=";", low_memory=False)
        ts = pd.read_csv(self.config[const.OBSERVED], sep=";", low_memory=False)
        self.data = reform.group_by_station(ts=ts, stations=self.stations)
        return self

    def basic(self):
        """
            Create a basic feature set from pollutant time series, per hour
            x: (time, longitude, latitude, pollutant values of t:t+n)
            y: (pollutant values of t+n:t+n+m)
        :return:
        """
        pollutants = ['PM10']  # ['PM2.5', 'PM10', 'O3']
        # columns = ['forecast', 'actual', 'station', 'pollutant']
        features = list()
        start_time = time.time()
        stations = self.stations.to_dict(orient='index')
        for _, s_info in stations.items():
            if s_info[const.PREDICT] != 1: continue
            s_data = self.data[s_info[const.ID]]
            s_time = pd.to_datetime(s_data[const.TIME], format=const.T_FORMAT, utc=True).tolist()
            for pollutant in pollutants:
                t, x, y = reform.split_dual(
                    time=s_time, value=s_data[pollutant].tolist(), unit_x=self.hour_x, unit_y=self.hour_y)
                # dayofweek = [time.dayofweek +  for time in enumerate(t)]
                loc = [[s_info[const.LONG], s_info[const.LAT]]] * len(t)  # location of station
                feature_set = [[t]+loc+x+y for t, loc, x, y in zip(t, loc, x, y)]
                features.extend(feature_set)
        # set name for columns
        columns = [const.TIME, const.LONG, const.LAT]
        columns.extend(['x' + str(i) for i in range(0, self.hour_x)])
        columns.extend(['y' + str(i) for i in range(0, self.hour_y)])
        self.features = pd.DataFrame(data=features, columns=columns)
        print('Basic features generated in', time.time() - start_time, 'secs')
        return self

    def sample(self, count):
        self.features = self.features.sample(n=count)
        return self

    def save(self):
        """
            Save the extracted features to file
        :return:
        """
        util.write(pd.DataFrame(data=self.features), address=self.config[const.FEATURES])


if __name__ == "__main__":
    addresses = settings.config[const.DEFAULT]
    fg = FeatureGenerator({
        const.OBSERVED: addresses[const.BJ_OBSERVED],
        const.STATIONS: addresses[const.BJ_STATIONS],
        const.FEATURES: addresses[const.BJ_FEATURES]
    }, hour_x=48, hour_y=48)
    fg.load().basic().sample(75000).save()
    print("Done!")
