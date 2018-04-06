import settings, const
from src import util
import pandas as pd


class DataSet:
    def __init__(self):
        self.data = {}
        self.stations = {}

    def load(self, from_time='0000-00-00 00:00:00', to_time='9999-01-01 00:00:00'):
        """
        Load data-set, do some pre-processing
        :param from_time: start of data time
        :param to_time: end of data time
        :return:
        """
        # access default configurations
        config = settings.config[const.DEFAULT]

        # read csv files
        data = pd.read_csv(config[const.CLEAN_DATA], sep=";", low_memory=False)

        # discard report of stations that do not report air quality
        data = data.loc[data['aq'] == 1, :]

        # filter a specific time interval
        data['utc_time'] = pd.to_datetime(data['utc_time'])
        data = util.filter_by_time(data, time_key='utc_time', from_time=from_time, to_time=to_time)

        # list of stations to iterate and predict
        stations = data.drop_duplicates(subset=[const.STATION_ID])[const.STATION_ID].tolist()

        # assign to class variables
        self.data = data
        self.stations = stations

        return self

    def get_pollutant(self, pollutant):
        """
        Get data for given pollutant, separated per station. with filled missing values
        :param pollutant:
        :return:
        """
        # put each station in a separate data-frame
        pollutants = {}
        for station in self.stations:
            # extract pollutants of specific station
            pollutants[station] = self.data.loc[self.data[const.STATION_ID] == station
            , ['utc_time', pollutant]].reset_index(drop=True)
            pollutants[station]['is_nan'] = pollutants[station][pollutant].isnull()
            # fill missing values with nearest neighbors (same column)
            util.fill(pollutants[station][pollutant], inplace=True)
        return pollutants
