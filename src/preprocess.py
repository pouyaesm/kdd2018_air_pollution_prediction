import const
from src import util
import pandas as pd


class PreProcess:

    def __init__(self, config):
        self.config = config  # location of input/output files
        self.obs = pd.DataFrame()  # merged observed air quality and meteorology data per station
        self.missing = pd.DataFrame()  # indicator of missing values in observed data-frame
        self.stations = pd.DataFrame()  # all stations with their attributes such as type and position

    def fill(self):
        """
            Fill missing value as the average of two nearest values in the time-line
            Per station
        :return:
        """
        columns = ['PM2.5', 'PM10', 'O3', 'NO2', 'CO', 'SO2',
                   'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']
        for _, column in enumerate(columns):
            if column not in self.obs.columns:
                continue  # go to next column
            for index, station in enumerate(self.stations[const.ID]):
                selector = self.obs[const.ID] == station
                station_ts = self.obs.ix[selector, column]
                if station_ts.isnull().all():
                    continue  # no value to fill the missing ones!
                self.obs.loc[selector, column] = util.fill(station_ts)
            print(column, 'missing values filled')

        return self
