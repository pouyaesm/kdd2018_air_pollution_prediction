import const
import settings
from src import util
import requests
import io
import pandas as pd
import numpy as np
import time


class PreProcessGrid:

    def __init__(self, cfg):
        self.config = cfg  # location of input/output files
        self.stations = pd.DataFrame()  # all stations with their attributes such as type and position
        self.grids = pd.DataFrame()  # id and location of grids

    def process(self):
        """
            Load and PreProcess the data
        :return:
        """
        # Read observed data and stations
        obs = pd.read_csv(self.config[const.OBSERVED], sep=';', low_memory=False)
        stations = pd.read_csv(self.config[const.STATIONS], sep=';')

        # Read grid data by chunk to avoid low memory exception
        tp = pd.read_csv(self.config[const.GRID_DATA], low_memory=False, iterator=True, chunksize=400000)
        grid_ts = pd.concat(tp, ignore_index=True)
        # grid_ts = tp.read(nrows=10000)  # for fast tests
        grid_ts.rename(columns={'stationName': const.GID, 'wind_speed/kph': const.WSPD}, inplace=True)

        # Append grid live loaded offline
        grid_live = pd.read_csv(self.config[const.GRID_LIVE], sep=';', low_memory=False)
        grid_ts = grid_ts.append(grid_live, ignore_index=True, verify_integrity=True)

        # Number of rows having missing value
        print('Rows with missing values:', np.count_nonzero(grid_ts[const.WSPD].isna()))

        # Extract grid ids and locations
        grids = grid_ts.groupby(const.GID, as_index=False) \
            .agg({const.LONG: 'first', const.LAT: 'first'})

        # Find distance of each station to each grid
        close_grids = dict()
        stations[const.GID] = ""
        for i, station in stations.iterrows():
            closest = grids.apply(
                lambda row: np.sqrt(
                    np.square(station[const.LONG] - row[const.LONG]) +
                    np.square(station[const.LAT] - row[const.LAT])
                ), axis=1)
            stations.loc[i, const.GID] = grids.ix[closest.idxmin(), const.GID]
        self.stations = stations
        return self

    def get_live(self):
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

    def fetch_save_all_live(self):
        grid_live = self.get_live()
        grid_live.to_csv(self.config[const.GRID_LIVE], sep=';', index=False)
        print('Data fetched and saved.')
        return self

    def save(self):
        """
            Save pre-processed data to files given in config
        :return:
        """
        # Write pre-processed data to csv file
        self.grids.to_csv(self.config[const.GRIDS], sep=';', index=False)
        self.stations.to_csv(self.config[const.STATIONS], sep=';', index=False)
        print('Data saved.')
        return self


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]
    pre_process_bj = PreProcessGrid({
        const.OBSERVED: config[const.BJ_OBSERVED],
        const.STATIONS: config[const.BJ_STATIONS],
        const.GRID_DATA: config[const.BJ_GRID_DATA],
        const.GRID_LIVE: config[const.BJ_GRID_LIVE],
        const.GRIDS: config[const.BJ_GRIDS]
    }).process().save()
    print("Done!")
