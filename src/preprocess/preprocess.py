import const
from src import util
import pandas as pd
import numpy as np
import time
from datetime import datetime
from datetime import timedelta
import io
import os


class PreProcess:

    def __init__(self, config=None):
        self.config = config  # location of input/output files
        self.obs = pd.DataFrame()  # merged observed air quality and meteorology data per station
        self.missing = pd.DataFrame()  # indicator of missing values in observed data-frame
        self.stations = pd.DataFrame()  # all stations with their attributes such as type and position
        self.grids = pd.DataFrame()  # information per weather grid

    def initialize(self, observed, stations):
        """
            Set observed and stations data to be used for filling
        :param observed:
        :param stations:
        :return:
        """
        self.obs = observed
        self.stations = stations
        return self

    def get_forecast_grid(self, city_code):
        """
        :param city_code: 'bj' or 'ld'
        :return:
        :rtype: pandas.DataFrame
        """
        template = "http://kdd.caiyunapp.com/competition/forecast/%s/2018-%s-%s-%s/2k0d1d8"
        dt = datetime.utcnow()
        while True:
            # url prepared for a given UTC time
            url = template % (city_code, dt.month, dt.day, dt.hour)
            forecast = pd.read_csv(io.StringIO(util.download(url)), error_bad_lines=False)
            if len(forecast) > 0:
                if 'id' not in forecast.columns:
                    return None     # API returned error
                forecast.rename(columns={'forecast_time': const.TIME, 'station_id': const.GID}, inplace=True)
                forecast.drop(columns=['id'], inplace=True)
                print(' Forecast grid data fetched from', dt.strftime(const.T_FORMAT))
                break
            else:  # if no data found at current UTC time try one hour before until data is available
                dt = dt - timedelta(hours=1)
        return forecast

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

    def fill(self, observed=None, stations=None, max_interval=0):
        """
            Fill missing value as the average of two nearest values in the time-line
            Per station
            Assumption: observed data must be sorted by station then by time
        :return:
        """
        # Can bypass load and pre-process and use the filling values directly
        if observed is not None:
            self.obs = observed
        if stations is not None:
            self.stations = stations
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

        # Keep grids for saving
        self.grids = grids.sort_values(by=[const.GID], ascending=True)

        return self

    def append_grid(self):
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
                               chunksize=1500000, float_precision='round_trip')

        def merge(observed, grid_chunk):
            # Join observed * station_grid * grid
            grid_chunk[const.TIME] = pd.to_datetime(grid_chunk[const.TIME], utc=True)
            observed = observed.merge(right=grid_chunk, how='left', on=[const.GID, const.TIME],
                                      suffixes=['_obs', '_grid'])
            # Merge observed and grid shared columns into one main column
            observed = util.merge_columns(observed, main='_obs', auxiliary='_grid')
            return observed

        # Merge grid forecast data with observed
        grid_forecast = pd.read_csv(self.config[const.GRID_FORECAST], sep=';', low_memory=False)
        # Create a table of (station_id, time, grid_id)
        # since observed data has no time corresponding to forecast grid data
        station_ids = list()
        grid_ids = list()
        times = list()
        for _, s_info in self.stations.iterrows():
            for t in grid_forecast[const.TIME].unique():
                station_ids.append(s_info[const.ID])
                grid_ids.append(s_info[const.GID])
                times.append(t)
        obs_forecast = pd.DataFrame(
            data={const.ID: station_ids, const.TIME: times, const.GID: grid_ids},
            columns=self.obs.columns)
        obs_forecast[const.TIME] = pd.to_datetime(obs_forecast[const.TIME], utc=True)
        obs_forecast = merge(obs_forecast, grid_forecast)
        self.obs = self.obs.append(other=obs_forecast, ignore_index=True)

        # Append grid live loaded offline
        grid_live = pd.read_csv(self.config[const.GRID_LIVE], sep=';', low_memory=False)
        self.obs = merge(self.obs, grid_live)

        # Merge historical grid data chunks with observed data
        for i, chunk in enumerate(iterator):
            print(' merge grid chunk {c}..'.format(c=i + 1))
            # clean historical values
            chunk.rename(columns={'stationName': const.GID, 'wind_speed/kph': const.WSPD}, inplace=True)
            self.clean_weather(chunk)
            self.obs = merge(self.obs, chunk)

        # remove possible duplicates due to forecast-live-history data overlap
        self.obs.drop_duplicates(subset=[const.ID, const.TIME])

        # Drop position and grid_id in observed data
        self.obs.drop(columns=[const.LONG, const.LAT, const.GID], inplace=True)

        # Sort data
        self.sort()

        return self

    def get_live_grid(self, time_from: datetime):
        """
            Load live observed data from KDD APIs
        :return:
        """
        url = self.config[const.GRID_URL] % (time_from.month, time_from.day, time_from.hour)
        grid_live = pd.read_csv(io.StringIO(util.download(url)))
        print('Live Grid has been read, count:', len(grid_live))
        # Rename fields to be compatible with offline data
        grid_live.rename(columns={
            const.ID: const.GID, 'time': const.TIME}, inplace=True
        )
        grid_live.drop(columns=['id'], inplace=True)

        return grid_live

    def fetch_save_live_grid(self, city_code):
        # read live data with 3 days overlap to have least missing values
        saved = None
        if os.path.isfile(self.config[const.GRID_LIVE]):
            saved = pd.read_csv(self.config[const.GRID_LIVE], sep=';', low_memory=False)
            latest_time_saved = pd.to_datetime(saved[const.TIME]).max() - timedelta(days=2)
        else:
            latest_time_saved = datetime(year=2018, month=1, day=1)

        grid_live = self.get_live_grid(time_from=latest_time_saved)
        if saved is not None:
            grid_live = grid_live.append(other=saved, ignore_index=True, verify_integrity=True)
            size = len(grid_live.index)
            # Keep the live values on duplicate to override the forecast saved previously with real
            grid_live.drop_duplicates(subset=[const.GID, const.TIME], inplace=True)
            print(' %d records overlap between live and saved' % (size - len(grid_live.index)))

        grid_forecast = self.get_forecast_grid(city_code=city_code)

        # sort dataframes for readability
        grid_live.sort_values(by=[const.GID, const.TIME], inplace=True)
        if grid_forecast is not None:
            grid_forecast.sort_values(by=[const.GID, const.TIME], inplace=True)

        # clean live and forecast values
        self.clean_weather(grid_live)
        if grid_forecast is not None:
            self.clean_weather(grid_forecast)

        grid_live.to_csv(self.config[const.GRID_LIVE], sep=';', index=False)
        if grid_forecast is not None:
            grid_forecast.to_csv(self.config[const.GRID_FORECAST], sep=';', index=False)
        else:   # save live grid as forecast to avoid further failures
            grid_live.to_csv(self.config[const.GRID_FORECAST], sep=';', index=False)
        print(' Grid live / forecast data fetched and saved.')
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
        self.grids.to_csv(self.config[const.GRIDS], sep=';', index=False)
        print('Data saved.')
        return self

    def get_observed(self):
        return self.obs

    def get_stations(self):
        return self.stations

    @staticmethod
    def clean_weather(df: pd.DataFrame):
        """
        :param df:
        :return:
        :rtype: pandas.DataFrame
        """
        # Change invalid values to NaN
        df.loc[df[const.TEMP] > 100, const.TEMP] = np.nan  # max value ~ 40 c
        df.loc[df[const.PRES] > 10000, const.PRES] = np.nan  # max value ~ 2000
        df.loc[df[const.WSPD] > 100, const.WSPD] = np.nan  # max value ~ 15 m/s
        df.loc[df[const.WDIR] > 360, const.WDIR] = -1  # no wind, value = [0, 360] degree
        df.loc[df[const.HUM] > 100, const.HUM] = np.nan  # value = [0, 100] percent
        return df