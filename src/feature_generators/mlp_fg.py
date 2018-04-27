import const
import settings
from src.preprocess import reform, times
from src import util
import time
import pandas as pd


class FeatureGenerator:

    def __init__(self, cfg, hour_x, hour_y):
        """
        :param cfg:
        :param hour_x: number of hour values of x (input) per sample
        :param hour_y: number of hour values of y (output) per sample
        """
        self.config = cfg
        self.data = pd.DataFrame()  # time series data per station
        self.stations = pd.DataFrame  # stations of time series
        self.features = pd.DataFrame()  # extracted features
        self.hour_x = hour_x
        self.hour_y = hour_y

    def load(self):
        # load_model data
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
        pollutant = self.config[const.POLLUTANT]  # ['PM2.5', 'PM10', 'O3']
        # columns = ['forecast', 'actual', 'station', 'pollutant']
        features = list()
        # labels = list()
        start_time = time.time()
        stations = self.stations.to_dict(orient='index')
        for _, s_info in stations.items():
            if s_info[const.PREDICT] != 1: continue
            station_id = s_info[const.ID]
            s_data = self.data[station_id]
            if s_data[const.TEMP].isnull().values.any(): continue
            s_time = pd.to_datetime(s_data[const.TIME], format=const.T_FORMAT, utc=True).tolist()
            first_x = self.hour_x - 1
            last_x = len(s_time) - self.hour_y - 1
            # temperature of last 7 days every 3 hours
            # temp_h6_28 = times.split(time=s_time, value=s_data[const.TEMP].tolist(),
            #                          hours=6, step=28, skip=first_x_end)
            # wspd_h6_28 = times.split(time=s_time, value=s_data[const.WSPD].tolist(),
            #                          hours=6, step=28, skip=first_x_end)
            # dayofweek = [time.dayofweek +  for time in enumerate(t)]
            # location of station
            # loc = [[s_info[const.LONG], s_info[const.LAT]]] * (len(s_time) - self.hour_x)
            s_value = s_data[pollutant].tolist()
            t, value = reform.split(time=s_time, value=s_value, step=self.hour_x)
            # v_h6_28 = times.split(time=s_time, value=s_value, hours=6, step=28, skip=first_x_end)
            # next hour_y values to be predicted
            label = times.split(time=s_time, value=s_value, group_hours=1,
                                step=self.hour_y, region=(first_x + 1, last_x + 1))
            sid = [station_id] * (len(s_time) - self.hour_x)
            feature_set = [[s]+[t]+v+l for s, t, v, l
                           in zip(sid, t, value, label)]
            features.extend(feature_set)
        # set name for columns
        columns = [const.ID, const.TIME] # [const.TIME, const.LONG, const.LAT]
        # columns.extend(['h6_' + str(i) for i in range(0, 28)])
        # columns.extend(['temp' + str(i) for i in range(0, 28)])
        # columns.extend(['wspd' + str(i) for i in range(0, 28)])
        columns.extend(['v' + str(i) for i in range(0, self.hour_x)])
        columns.extend(['l' + str(i) for i in range(0, self.hour_y)])
        self.features = pd.DataFrame(data=features, columns=columns)
        print(len(self.features.index), 'feature vectors generated in',
              time.time() - start_time, 'secs')
        return self

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
        util.write(self.features, address=self.config[const.FEATURE])
        print(len(self.features.index), 'feature vectors are written to file')


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]
    pollutant = 'PM10'
    features_bj = config[getattr(const, 'BJ_' + pollutant.replace('.', '') + '_')] + 'mlp_features.csv'
    features_ld = config[getattr(const, 'LD_' + pollutant.replace('.', '') + '_')] + 'mlp_features.csv'
    config_bj = {
        const.OBSERVED: config[const.BJ_OBSERVED],
        const.STATIONS: config[const.BJ_STATIONS],
        const.FEATURE: features_bj,
        const.POLLUTANT: pollutant
    }
    config_ld = {
        const.OBSERVED: config[const.LD_OBSERVED],
        const.STATIONS: config[const.LD_STATIONS],
        const.FEATURE: features_ld,
        const.POLLUTANT: pollutant
    }
    fg = FeatureGenerator(config_bj, hour_x=48, hour_y=48)
    fg.load().basic().dropna().sample(100000).save()
    fg = FeatureGenerator(config_ld, hour_x=48, hour_y=48)
    fg.load().basic().dropna().save()
    print("Done!")
