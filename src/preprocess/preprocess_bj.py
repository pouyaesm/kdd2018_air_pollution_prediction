import settings
import const
import io
import pandas as pd
import numpy as np
import requests
from src import util
from src.preprocess.preprocess import PreProcess


class PreProcessBJ(PreProcess):

    def __init__(self, config):
        super(PreProcessBJ, self).__init__(config)
        self.config = config  # location of input/output files
        self.obs = pd.DataFrame()  # merged observed air quality and meteorology data per station
        self.missing = pd.DataFrame()  # indicator of missing values in observed data-frame
        self.stations = pd.DataFrame()  # all stations with their attributes such as type and position

    def get_live(self):
        """
            Load live observed data from KDD APIs
        :return:
        """
        aq_url = "https://biendata.com/competition/airquality/bj/2018-02-01-0/2018-06-01-0/2k0d1d8"
        meo_url = "https://biendata.com/competition/meteorology/bj/2018-02-01-0/2018-06-01-0/2k0d1d8"
        aq_live = pd.read_csv(io.StringIO(util.download(aq_url)))
        print('Live aQ has been read, count:', len(aq_live))
        meo_live = pd.read_csv(io.StringIO(util.download(meo_url)))
        print('Live meO has been read, count:', len(meo_live))

        # Make live data columns compatible with offline data
        aq_live.rename(columns={col: col.split('_Concentration')[0] for col in aq_live.columns}, inplace=True)
        aq_live.rename(columns={'time': const.TIME, 'PM25': 'PM2.5'}, inplace=True)
        aq_live.drop(columns=['id'], inplace=True)
        meo_live.rename(columns={'time': const.TIME}, inplace=True)
        meo_live.drop(columns=['id'], inplace=True)

        return aq_live, meo_live

    def fetch_save_live(self):
        aq_live, meo_live = self.get_live()
        aq_live.to_csv(self.config[const.AQ_LIVE], sep=';', index=False)
        meo_live.to_csv(self.config[const.MEO_LIVE], sep=';', index=False)
        return self

    def process(self):
        """
            Load and PreProcess the data
        :return:
        """
        # Read two parts of beijing observed air quality data
        aq = pd.read_csv(self.config[const.AQ], low_memory=False) \
            .append(pd.read_csv(self.config[const.AQ_REST], low_memory=False)
                    , ignore_index=True, verify_integrity=True)
        # Rename stationId to station_id in quality table (the same as air weather table)
        aq.rename(columns={'stationId': const.ID}, inplace=True)
        # Load and append air quality live data
        aq = aq.append(pd.read_csv(self.config[const.AQ_LIVE], sep=';', low_memory=False)
                , ignore_index=True, verify_integrity=True)
        # drop possible overlapped duplicates
        aq.drop_duplicates(subset=[const.ID, const.TIME], inplace=True)

        # Read beijing station meteorology data, append live data
        meo = pd.read_csv(self.config[const.MEO], low_memory=False)\
            .append(pd.read_csv(self.config[const.MEO_LIVE], sep=';', low_memory=False)
                                , ignore_index=True, verify_integrity=True)
        # drop possible overlapped duplicates
        meo.drop_duplicates(subset=[const.ID, const.TIME], inplace=True)

        # Read type and position of air quality stations
        aq_stations = pd.read_csv(self.config[const.AQ_STATIONS], low_memory=False)

        # remove _aq, _meo postfixes from station ids
        aq[const.ID] = aq[const.ID].str.replace('_.*', '')  # name_aq -> name
        meo[const.ID] = meo[const.ID].str.replace('_.*', '')  # name_meo -> name

        # Convert string datetime to datetime object
        aq[const.TIME] = pd.to_datetime(aq[const.TIME], utc=True)
        meo[const.TIME] = pd.to_datetime(meo[const.TIME], utc=True)

        # Change invalid values to NaN
        meo.loc[meo[const.TEMP] > 100, const.TEMP] = np.nan  # max value ~ 40 c
        meo.loc[meo[const.PRES] > 10000, const.PRES] = np.nan  # max value ~ 2000
        meo.loc[meo[const.WSPD] > 100, const.WSPD] = np.nan  # max value ~ 15 m/s
        meo.loc[meo[const.WDIR] > 360, const.WDIR] = np.nan  # value = [0, 360] degree
        meo.loc[meo[const.HUM] > 100, const.HUM] = np.nan  # value = [0, 100] percent

        # Change negative reports of pollutants to NaN
        pollutants = [const.PM25, const.PM10, const.O3]
        for pollutant in pollutants:
            negatives = aq[pollutant] < 0
            aq.loc[negatives, pollutant] = np.nan
            print('Negative {p} reports: {c}'.format(p=pollutant, c=np.sum(negatives)))

        # Merge air quality and weather data based on stationId-timestamp
        self.obs = aq.merge(meo, how='outer', on=[const.ID, const.TIME], suffixes=['_aq', '_meo'])

        # Build a table for stations data and remove their attributes from other tables
        stations = self.obs.groupby('station_id', as_index=False) \
            .agg({'longitude': 'first', 'latitude': 'first'})
        stations = stations.merge(aq_stations, how='outer', on=[const.ID], suffixes=['', '_aq'])
        stations = util.fillna(stations, target=['longitude', 'latitude'], source=['longitude_aq', 'latitude_aq'])
        # Stations that has type are air quality ones and need to be predicted
        stations[const.PREDICT] = (1 - stations[const.S_TYPE].isna()).astype(int)

        self.stations = util.drop_columns(stations, end_with='_aq')

        # Remove station position from time series
        self.obs.drop(['longitude', 'latitude'], axis=1)

        # Remove reports with no station id
        self.obs.dropna(subset=[const.ID], inplace=True)

        # Sort data
        self.sort()

        return self


if __name__ == "__main__":
    pre_process = PreProcessBJ(settings.config[const.DEFAULT])
    pre_process.process()
    pre_process.fill()
    print('No. observed rows:', len(pre_process.obs))
    print('No. stations:', len(pre_process.stations),
          ', only weather:', pre_process.stations['station_type'].count())
    pre_process.save()
    print("Done!")
