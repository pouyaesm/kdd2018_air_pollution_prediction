import const
from src import util
import pandas as pd
import time

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
        # Reset time series indices to ensure a closed interval
        self.obs.reset_index(drop=True, inplace=True)

        columns = ['PM2.5', 'PM10', 'O3', 'NO2', 'CO', 'SO2',
                   'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']

        start_time = time.time()
        for _, station in enumerate(self.stations[const.ID]):
            selector = self.obs[const.ID] == station
            for _, column in enumerate(columns):
                if column not in self.obs.columns:
                    continue  # go to next column
                station_ts = self.obs.ix[selector, column]
                if station_ts.isnull().all():
                    continue  # no value to fill the missing ones!
                self.obs.loc[selector, column] = util.fill(station_ts, inplace=True)

        print('missing values filled in', time.time() - start_time, 'secs')

        return self
