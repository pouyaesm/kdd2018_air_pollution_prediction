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

    def sort(self):
        """
            Sort data first based on station ids (alphabetically), then by time ascending
            Using inplace creates a warning!
        :return:
        """
        self.stations = self.stations.sort_values(const.ID, ascending=True)
        self.obs = self.obs.sort_values([const.ID, const.TIME], ascending=True)
        return self

    def fill(self, max_interval=0):
        """
            Fill missing value as the average of two nearest values in the time-line
            Per station
            Assumption: observed data must be sorted by station then by time
        :return:
        """
        # Reset time series indices to ensure a closed interval
        self.obs.reset_index(drop=True, inplace=True)

        # mark missing values
        self.missing = self.obs.isna().astype(int)

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
                self.obs.loc[selector, column] \
                    = util.fill(station_ts, max_interval=max_interval, inplace=True)

        print('Missing values filled in', time.time() - start_time, 'secs')

        return self

    def _relate_observed_to_grid(self):
        """
            Add grid_id to each row of observed station data
        :return:
        """
        # Read grid data by chunk to avoid low memory exception
        grid_ts = pd.read_csv(self.config[const.GRID_DATA], low_memory=False,
                               iterator=True, float_precision='round_trip').read(1000)
        grid_ts.rename(columns={'stationName': const.GID}, inplace=True)
        # Extract grid ids and locations
        grids = grid_ts.groupby(const.GID, as_index=False) \
            .agg({const.LONG: 'first', const.LAT: 'first'})

        # Find closest grid to each station
        for i, station in self.stations.iterrows():
            closest = grids.apply(
                lambda row: np.sqrt(
                    np.square(station[const.LONG] - row[const.LONG]) +
                    np.square(station[const.LAT] - row[const.LAT])
                ), axis=1)
            self.stations.loc[i, const.GID] = grids.ix[closest.idxmin(), const.GID]

        # Relate observed data to grid ids
        self.obs = self.obs.merge(right=self.stations[[const.ID, const.GID]], on=const.ID)

        return self

    def append_grid(self, include_history=False):
        """
            Load grid offline and live data, append the complementary weather info
            to observed time series
        :return:
        """
        if len(self.obs.index) == 0 or len(self.stations.index) == 0:
            raise ValueError("Observed and station data must be prepared first")

        # Relate stations to closest grid for joining grid data to observed data
        self._relate_observed_to_grid()

        # Read huge grid data part by part
        iterator = pd.read_csv(self.config[const.GRID_DATA], iterator=True, low_memory=False,
                               chunksize=2000000, float_precision='round_trip')

        def merge(grid_chunk):
            # Join observed * station_grid * grid
            grid_chunk.rename(columns={'stationName': const.GID, 'wind_speed/kph': const.WSPD}, inplace=True)
            grid_chunk[const.TIME] = pd.to_datetime(grid_chunk[const.TIME], utc=True)
            self.obs = self.obs.merge(right=grid_chunk, how='left', on=[const.GID, const.TIME],
                                      suffixes=['_obs', '_grid'])
            # Merge observed and grid shared columns into one main column
            self.obs = util.merge_columns(self.obs, main='_obs', auxiliary='_grid')

        for i, chunk in enumerate(iterator):
            print(' merge grid chunk {c}..'.format(c=i + 1))
            merge(chunk)

        # Append grid live loaded offline
        grid_live = pd.read_csv(self.config[const.GRID_LIVE], sep=';', low_memory=False)
        merge(grid_live)

        # Drop position and grid_id in observed data
        self.obs.drop(columns=[const.LONG, const.LAT, const.GID], inplace=True)

        # Sort data
        self.sort()

        return self

    def get_live_grid(self):
        """
            Load live observed data from KDD APIs
        :return:
        """
        grid_live = pd.read_csv(io.StringIO(util.download(self.config[const.GRID_URL])))
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

    def save(self, append=False):
        """
            Save pre-processed data to files given in config
        :return:
        """
        if append:
            # Read and append saved observed data
            self.obs = pd.read_csv(self.config[const.OBSERVED], sep=';', low_memory=True) \
                .append(other=self.obs, ignore_index=True, verify_integrity=True)
            self.obs.drop_duplicates(subset=[const.ID, const.TIME], inplace=True)

            # Sort data
            self.sort()

            # mark missing values
            self.missing = self.obs.isna().astype(int)

        # Write pre-processed data to csv file
        util.write(self.obs, self.config[const.OBSERVED])
        util.write(self.missing, self.config[const.OBSERVED_MISSING])
        self.stations.to_csv(self.config[const.STATIONS], sep=';', index=False)
        if len(self.grids.index) > 0:
            self.grids.to_csv(self.config[const.GRIDS], sep=';', index=False)
        print('Data saved.')
        return self
