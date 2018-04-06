import settings
import const
import io
import pandas as pd
import numpy as np
import requests
from src import util


class PreProcessBJ:

    def __init__(self, config):
        self.config = config  # location of input/output files
        self.obs = pd.DataFrame()  # merged observed air quality and meteorology data per station
        self.missing = pd.DataFrame()  # indicator of missing values in observed data-frame
        self.stations = pd.DataFrame()  # all stations with their attributes such as type and position

    @staticmethod
    def get_live():
        """
            Load live observed data from KDD APIs
        :return:
        """
        aq_url = "https://biendata.com/competition/airquality/bj/2018-03-01-0/2018-06-01-0/2k0d1d8"
        meo_url = "https://biendata.com/competition/meteorology/bj/2018-03-01-0/2018-06-01-0/2k0d1d8"
        aq_live = pd.read_csv(io.StringIO(requests.get(aq_url).content.decode('utf-8')))
        print('Live aQ has been read, count:', len(aq_live))
        meo_live = pd.read_csv(io.StringIO(requests.get(meo_url).content.decode('utf-8')))
        print('Live meO has been read, count:', len(meo_live))
        return aq_live, meo_live

    def process(self):
        """
            Load and PreProcess the data
        :return:
        """
        # Read two parts of beijing observed air quality data
        aq = pd.read_csv(self.config[const.BJ_AQ], low_memory=False) \
            .append(pd.read_csv(self.config[const.BJ_AQ_REST], low_memory=False)
                    , ignore_index=True, verify_integrity=True)

        # Real live data from APIs
        aq_live, meo_live = self.get_live()

        # Rename stationId to station_id in quality table (the same as air weather table)
        aq.rename(columns={'stationId': const.ID}, inplace=True)

        # Make live data columns compatible with offline data
        aq_live.rename(columns={col: col.split('_Concentration')[0] for col in aq_live.columns}, inplace=True)
        aq_live.rename(columns={'time': const.TIME, 'PM25': 'PM2.5'}, inplace=True)
        aq_live.drop(columns=['id'], inplace=True)
        meo_live.rename(columns={'time': const.TIME}, inplace=True)
        meo_live.drop(columns=['id'], inplace=True)

        # Append live data to offline data, and drop possible overlapped duplicates
        aq = aq.append(aq_live, ignore_index=True, verify_integrity=True)
        aq.drop_duplicates(subset=[const.ID, const.TIME], inplace=True)

        # Read beijing station meteorology data, append live data
        meo = pd.read_csv(self.config[const.BJ_MEO], low_memory=False)\
            .append(meo_live, ignore_index=True, verify_integrity=True)
        meo.drop_duplicates(subset=[const.ID, const.TIME], inplace=True)

        # Read type and position of air quality stations
        aq_stations = pd.read_csv(self.config[const.BJ_AQ_STATIONS], low_memory=False)

        # remove _aq, _meo postfixes from station ids
        aq[const.ID] = aq[const.ID].str.replace('_.*', '')  # name_aq -> name
        meo[const.ID] = meo[const.ID].str.replace('_.*', '')  # name_meo -> name

        # Convert string datetime to datetime object
        aq[const.TIME] = pd.to_datetime(aq[const.TIME], utc=True)
        meo[const.TIME] = pd.to_datetime(meo[const.TIME], utc=True)

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

        # mark missing values
        self.missing = self.obs.isna().astype(int)

        return self

    def save(self):
        """
            Save pre-processed data to files given in config
        :return:
        """
        # Write pre-processed data to csv file
        util.write(self.obs, self.config[const.BJ_OBSERVED])
        util.write(self.missing, self.config[const.BJ_OBSERVED_MISS])
        # self.missing.to_csv(, sep=';', index=False)
        self.stations.to_csv(self.config[const.BJ_STATIONS], sep=';', index=False)
        print('Data saved.')


if __name__ == "__main__":
    pre_process = PreProcessBJ(settings.config[const.DEFAULT])
    pre_process.process()
    print('No. observed rows:', len(pre_process.obs))
    print('No. stations:', len(pre_process.stations),
          ', only weather:', pre_process.stations['station_type'].count())
    pre_process.save()
    print("Done!")
