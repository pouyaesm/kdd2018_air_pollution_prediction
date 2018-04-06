import settings
import const
import pandas as pd
import numpy as np
from src import util


class PreProcess:

    def __init__(self, config):
        self.config = config  # location of input/output files
        self.obs = pd.DataFrame()  # merged observed air quality and meteorology data per station
        self.stations = pd.DataFrame()  # all stations with their attributes such as type and position

    def load(self):
        """
            Load and PreProcess the data
        :return:
        """
        # Read csv files
        # read two parts of beijing observed air quality data
        aq = pd.read_csv(self.config[const.BJ_AQ], low_memory=False) \
            .append(pd.read_csv(self.config[const.BJ_AQ_REST], low_memory=False), ignore_index=True)
        # read beijing station meteorology data
        meo = pd.read_csv(self.config[const.BJ_MEO], low_memory=False)
        # read type and position of air quality stations
        aq_stations = pd.read_csv(self.config[const.BJ_AQ_STATIONS], low_memory=False)

        # Rename stationId to station_id in quality table (the same as air weather table)
        aq.rename(columns={'stationId': const.ID}, inplace=True)
        # remove _aq, _meo postfixes from station ids
        aq[const.ID] = aq[const.ID].str.replace('_.*', '')  # name_aq -> name
        meo[const.ID] = meo[const.ID].str.replace('_.*', '')  # name_meo -> name

        # Change invalid values to NaN
        meo.loc[meo['temperature'] > 100, 'temperature'] = np.nan  # max value ~ 40 c
        meo.loc[meo['wind_speed'] > 100, 'wind_speed'] = np.nan  # max value ~ 15 m/s
        meo.loc[meo['wind_direction'] > 360, 'wind_direction'] = np.nan  # value = [0, 360] degree
        meo.loc[meo['humidity'] > 100, 'humidity'] = np.nan  # value = [0, 100] percent

        # Merge air quality and weather data based on stationId-timestamp
        self.obs = aq.merge(meo, how='outer', on=[const.ID, const.TIME], suffixes=['_aq', '_meo'])

        # Build a table for stations data and remove their attributes from other tables
        stations = self.obs.groupby('station_id', as_index=False) \
            .agg({'longitude': 'first', 'latitude': 'first'})
        stations = stations.merge(aq_stations, how='outer', on=[const.ID], suffixes=['', '_aq'])
        stations = util.fillna(stations, target=['longitude', 'latitude'], source=['longitude_aq', 'latitude_aq'])
        self.stations = util.drop_columns(stations, end_with='_aq')
        # Remove station position from time series
        self.obs.drop(['longitude', 'latitude'], axis=1)

        # Remove reports with no station id
        self.obs.dropna(subset=[const.ID], inplace=True)

        # Sort data first based on station ids (alphabetically), then by time ascending
        self.stations.sort_values(const.ID, ascending=True)
        self.obs.sort_values([const.ID, const.TIME], ascending=True)

        return self

    def save(self):
        """
            Save pre-processed data to files given in config
        :return:
        """
        # Write pre-processed data to csv file
        self.obs.to_csv(self.config[const.BJ_OBSERVED], sep=';', index=False)
        self.stations.to_csv(self.config[const.BJ_STATIONS], sep=';', index=False)
        print('Data saved.')


if __name__ == "__main__":
    pre_process = PreProcess(settings.config[const.DEFAULT])
    pre_process.load()
    print('No. observed rows:', len(pre_process.obs))
    print('No. stations:', len(pre_process.stations),
          ', only weather:', pre_process.stations['station_type'].count())
    pre_process.save()
    print("Done!")
