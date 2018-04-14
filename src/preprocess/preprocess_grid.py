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
        self.grid = pd.DataFrame()  # merged observed air quality and meteorology data per station
        self.stations = pd.DataFrame()  # all stations with their attributes such as type and position

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
            'stationName': const.ID, 'time': const.TIME, 'wind_speed/kph': const.WSPD}, inplace=True
        )
        grid_live.drop(columns=['id'], inplace=True)

        return grid_live

    def fetch_save_all_live(self):
        grid_live = self.get_live()
        grid_live.to_csv(self.config[const.GRID_LIVE], sep=';', index=False)

    def process(self):
        """
            Load and PreProcess the data
        :return:
        """
        # Read grid data
        tp = pd.read_csv(self.config[const.GRID], low_memory=False, iterator=True, chunksize=400000)
        grid = pd.concat(tp, ignore_index=True)
        grid.rename(columns={'stationName': const.ID, 'wind_speed/kph': const.WSPD}, inplace=True)
        # Append grid live loaded offline
        grid_live = pd.read_csv(self.config[const.GRID_LIVE], sep=';', low_memory=False)
        grid = grid.append(grid_live, ignore_index=True, verify_integrity=True)

        stations = pd.read_csv(self.config[const.STATIONS])

        return self

    def save(self):
        """
            Save pre-processed data to files given in config
        :return:
        """
        # Write pre-processed data to csv file
        util.write(self.grid, self.config[const.BJ_OBSERVED])
        # self.missing.to_csv(, sep=';', index=False)
        self.stations.to_csv(self.config[const.BJ_STATIONS], sep=';', index=False)
        print('Data saved.')
        return self


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]
    pre_process_bj = PreProcessGrid({
        const.STATIONS: config[const.BJ_STATIONS],
        const.GRID: config[const.BJ_GRID],
        const.GRID_LIVE: config[const.BJ_GRID_LIVE]
    }).process().save()
    print("Done!")
