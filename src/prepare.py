import settings
import const
import pandas as pd
from src import util


# Prepare time series data for learning
class Prepare:

    def __init__(self, config):
        self.config = config  # path of required files for preparing
        self.ts = pd.DataFrame()  # time series data
        self.stations = pd.DataFrame()  # stations data
        self.missing = pd.DataFrame()  # index of missing values in the data

    def load(self):
        """
            Load pre-processed data
        :return:
        """
        self.ts = pd.read_csv(self.config[const.BJ_OBSERVED], delimiter=';', low_memory=False)
        self.stations = pd.read_csv(self.config[const.BJ_STATIONS], delimiter=';', low_memory=False)
        self.missing = self.ts.isna()
        return self

    def fill(self):
        """
            Fill missing value as the average of two nearest values in the time-line
            Per station
        :return:
        """
        for index, station in enumerate(self.stations[const.ID]):
            station_ts = self.ts.loc[self.ts[const.ID] == station]
            station_ts['PM2.5'].loc[:] = util.fill(station_ts['PM2.5'])
            print(station)

        return self

    def save(self):
        """
            Save pre-processed data to files given in config
        :return:
        """
        # Write pre-processed data to csv file
        util.write(self.ts, self.config[const.BJ_OBSERVED])
        util.write(self.missing, self.config[const.BJ_OBSERVED_MISS])
        # self.missing.to_csv(, sep=';', index=False)
        self.stations.to_csv(self.config[const.BJ_STATIONS], sep=';', index=False)
        print('Data saved.')


if __name__ == "__main__":
    prepare = Prepare(settings.config[const.DEFAULT])
    prepare.load().fill()
    print("Done!")


