import const
from src import util
import pandas as pd
import numpy as np
import time
import requests
import io


class PreProcess:

    def __init__(self, config):
        self.config = config  # location of input/output files
        self.obs = pd.DataFrame()  # merged observed air quality and meteorology data per station
        self.missing = pd.DataFrame()  # indicator of missing values in observed data-frame
        self.stations = pd.DataFrame()  # all stations with their attributes such as type and position
        self.grids = pd.DataFrame()  # information per weather grid

    def get_live(self):
        return None

    def fetch_save_live(self):
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

    def append_grid(self):
        """
            Load grid offline and live data, append the colplementary weather info
            to observed time series
        :return:
        """
        if len(self.obs.index) == 0 or len(self.stations.index) == 0:
            raise ValueError("Observed and station data must be prepared first")

        # Read grid data by chunk to avoid low memory exception
        iterator = pd.read_csv(self.config[const.GRID_DATA],
                               low_memory=False, iterator=True, chunksize=200000)
        grid_ts = pd.concat(iterator, ignore_index=True) \
            if not self.config[const.IS_TEST] else iterator.read(nrows=10000)  # sample for testing fast
        grid_ts.rename(columns={'stationName': const.GID, 'wind_speed/kph': const.WSPD}, inplace=True)

        # Append grid live loaded offline
        grid_live = pd.read_csv(self.config[const.GRID_LIVE], sep=';', low_memory=False)
        grid_ts = grid_ts.append(grid_live, ignore_index=True, verify_integrity=True)

        # Reformat observable time from compact to full, to be joinable with grid data
        self.obs[const.TIME] = pd.to_datetime(self.obs[const.TIME], format=const.T_FORMAT) \
            .dt.strftime(const.T_FORMAT_FULL)

        # Number of rows having missing value
        print('Grid rows with missing values:', np.count_nonzero(grid_ts[const.WSPD].isna()))

        # Extract grid ids and locations
        grids = grid_ts.loc[0:1000, :].groupby(const.GID, as_index=False) \
            .agg({const.LONG: 'first', const.LAT: 'first'})

        # Find closest grid to each station
        for i, station in self.stations.iterrows():
            closest = grids.apply(
                lambda row: np.sqrt(
                    np.square(station[const.LONG] - row[const.LONG]) +
                    np.square(station[const.LAT] - row[const.LAT])
                ), axis=1)
            self.stations.loc[i, const.GID] = grids.ix[closest.idxmin(), const.GID]

        # Join observed * station_grid * grid
        self.obs = self.obs.merge(right=self.stations[[const.ID, const.GID]], on=const.ID) \
            .merge(right=grid_ts, on=[const.GID, const.TIME], suffixes=['_obs', '_grid'])
        self.obs = util.merge_columns(self.obs, main='_obs', auxiliary='_grid')

        # Drop position and grid_id in observed data
        self.obs.drop(columns=[const.LONG, const.LAT, const.GID], inplace=True)

        return self

    def get_live_grid(self):
        """
            Load live observed data from KDD APIs
        :return:
        """
        grid_live = pd.read_csv(io.StringIO(
            requests.get(self.config[const.GRID_URL]).content.decode('utf-8')
        ))
        print('Live Grid has been read, count:', len(grid_live))
        # Rename fields to be compatible with offline data
        grid_live.rename(columns={
            const.ID: const.GID, 'time': const.TIME}, inplace=True
        )
        grid_live.drop(columns=['id'], inplace=True)

        return grid_live

    def fetch_save_live_grid(self):
        grid_live = self.get_live_grid()
        grid_live.to_csv(self.config[const.GRID_LIVE], sep=';', index=False)
        print('Grid data fetched and saved.')
        return self

    def save(self):
        """
            Save pre-processed data to files given in config
        :return:
        """
        # Write pre-processed data to csv file
        util.write(self.obs, self.config[const.OBSERVED])
        util.write(self.missing, self.config[const.OBSERVED_MISSING])
        self.stations.to_csv(self.config[const.STATIONS], sep=';', index=False)
        if len(self.grids.index) > 0:
            self.grids.to_csv(self.config[const.GRIDS], sep=';', index=False)
        print('Data saved.')
        return self
