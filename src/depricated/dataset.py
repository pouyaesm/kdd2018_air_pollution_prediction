import settings, const
from src import util
import pandas as pd


class DataSet:
    def __init__(self, config):
        self.obs = {}
        self.stations = {}
        self.config = config

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
        data = pd.read_csv(config[const.BJ_OBSERVED], sep=";", low_memory=False)
        self.stations = pd.read_csv(config[const.BJ_STATIONS], sep=";", low_memory=False)

        self.obs = data

        return self

    def fill_missing(self):
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


if __name__ == "__main__":
    pre_process = DataSet(settings.config[const.DEFAULT])
    print("Done!")
