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
        pollutants = [self.config[const.POLLUTANT]]  # ['PM2.5', 'PM10', 'O3']
        # columns = ['forecast', 'actual', 'station', 'pollutant']
        features = list()
        # labels = list()
        start_time = time.time()
        stations = self.stations.to_dict(orient='index')
        for _, s_info in stations.items():
            if s_info[const.PREDICT] != 1: continue
            s_data = self.data[s_info[const.ID]]
            if s_data[const.TEMP].isnull().values.any(): continue
            s_time = pd.to_datetime(s_data[const.TIME], format=const.T_FORMAT, utc=True).tolist()
            first_x_end = self.hour_x - 1
            # temperature of last 7 days every 3 hours
            temp_h6_28 = times.split(time=s_time, value=s_data[const.TEMP].tolist(),
                                     hours=6, step=28, skip=first_x_end)
            wspd_h6_28 = times.split(time=s_time, value=s_data[const.WSPD].tolist(),
                                     hours=6, step=28, skip=first_x_end)
            # dayofweek = [time.dayofweek +  for time in enumerate(t)]
            # location of station
            loc = [[s_info[const.LONG], s_info[const.LAT]]] * (len(s_time) - self.hour_x)
            for pollutant in pollutants:
                s_value = s_data[pollutant].tolist()
                t_h_48, v_h_48 = reform.split(time=s_time, value=s_value, step=self.hour_x)
                v_h6_28 = times.split(time=s_time, value=s_value, hours=6, step=28, skip=first_x_end)
                # next hour_y values to be predicted
                v_h_y = times.split(time=s_time, value=s_value, hours=1,
                                    step=self.hour_y, skip=first_x_end + self.hour_y)
                feature_set = [[t]+loc+h6+temp+wspd+h+y for t, loc, h6, temp, wspd, h, y
                               in zip(t_h_48, loc, v_h6_28, temp_h6_28, wspd_h6_28, v_h_48, v_h_y)]
                features.extend(feature_set)
        # set name for columns
        columns = [const.TIME, const.LONG, const.LAT]
        columns.extend(['h6_' + str(i) for i in range(0, 28)])
        columns.extend(['temp' + str(i) for i in range(0, 28)])
        columns.extend(['wspd' + str(i) for i in range(0, 28)])
        columns.extend(['h' + str(i) for i in range(0, self.hour_x)])
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
        util.write(self.features, address=self.config[const.FEATURES])


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]
    fg = FeatureGenerator({
        const.OBSERVED: config[const.BJ_OBSERVED],
        const.STATIONS: config[const.BJ_STATIONS],
        const.FEATURES: config[const.BJ_PM25_FEATURES],
        const.POLLUTANT: 'PM2.5'
    }, hour_x=48, hour_y=48)
    fg.load().basic().sample(75000).save()
    print("Done!")
