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

    def get_live(self):
        return None

    def fetch_save_all_live(self):
        return self

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
                station_ts = self.obs.loc[selector, column]
                if station_ts.isnull().all():
                    continue  # no value to fill the missing ones!
                self.obs.loc[selector, column] = util.fill(station_ts, inplace=True)

        print('Missing values filled in', time.time() - start_time, 'secs')

        return self

    def process_grid(self):
        """
            Load and PreProcess the grid data
        :return:
        """
        if len(self.obs.index) == 0 or len(self.stations.index) == 0:
            raise ValueError("Observed data must be prepared first")
        # # Read observed data and stations
        # obs = pd.read_csv(self.config[const.OBSERVED], sep=';', low_memory=False)
        # stations = pd.read_csv(self.config[const.STATIONS], sep=';')
        # Read grid data by chunk to avoid low memory exception
        tp = pd.read_csv(self.config[const.GRID_DATA], low_memory=False, iterator=True, chunksize=400000)
        # grid_ts = pd.concat(tp, ignore_index=True)
        grid_ts = tp.read(nrows=10000)  # for fast tests
        grid_ts.rename(columns={'stationName': const.ID, 'wind_speed/kph': const.WSPD}, inplace=True)
        # Append grid live loaded offline
        grid_live = pd.read_csv(self.config[const.GRID_LIVE], sep=';', low_memory=False)
        grid_ts = grid_ts.append(grid_live, ignore_index=True, verify_integrity=True)
        grids = grid_ts.groupby('station_id', as_index=False) \
            .agg({const.LONG: 'first', const.LAT: 'first'})
        for station_info in self.stations.itertuples():
            longitude = station_info.longitude
            latitude = station_info.latitude
        return self
