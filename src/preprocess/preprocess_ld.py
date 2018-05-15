import settings
import const
import io
import requests
import pandas as pd
from src import util
from src.preprocess.preprocess import PreProcess
import numpy as np


class PreProcessLD(PreProcess):

    def get_live(self):
        """
            Load live observed data from KDD APIs
        :return:
        """
        aq_url = "https://biendata.com/competition/airquality/ld/2018-02-01-0/2018-06-05-0/2k0d1d8"
        aq_live = pd.read_csv(io.StringIO(requests.get(aq_url).content.decode('utf-8')))
        print('Live aQ has been read, count:', len(aq_live))
        return aq_live

    def fetch_save_live(self):
        aq_live = self.get_live()
        aq_live.rename(columns={col: col.split('_Concentration')[0] for col in aq_live.columns}, inplace=True)
        aq_live.rename(columns={'time': const.TIME, 'PM25': 'PM2.5'}, inplace=True)
        aq_live.drop(columns=['id'], inplace=True)
        aq_live.to_csv(self.config[const.AQ_LIVE], sep=';', index=False)
        return self

    def process(self):
        """
            Load and PreProcess the data
        :return:
        """
        # Read two parts of london observed air quality data
        aq = pd.read_csv(self.config[const.AQ], low_memory=False)
        aq_rest = pd.read_csv(self.config[const.AQ_REST], low_memory=False)
        aq_live = pd.read_csv(self.config[const.AQ_LIVE], sep=';', low_memory=False)

        # Read type and position of stations
        aq_stations = pd.read_csv(self.config[const.AQ_STATIONS], low_memory=False)

        # Remove the index column (first) of aq
        aq.drop(aq.columns[0], axis=1, inplace=True)

        # Rename columns to conventional ones, and change 'value (ug/m3)' to 'value'
        aq.rename(columns={'MeasurementDateGMT': const.TIME}, inplace=True)
        aq.rename(columns={col: col.split(' ')[0] for col in aq.columns}, inplace=True)

        aq_rest.rename(columns={'Station_ID': const.ID, 'MeasurementDateGMT': const.TIME}, inplace=True)
        aq_rest.rename(columns={col: col.split(' ')[0] for col in aq_rest.columns}, inplace=True)
        # Drop last two completely null columns from aq_rest
        aq_rest.dropna(axis=1, how='all', inplace=True)

        # Append aq_rest and aq_live to aq
        aq = aq.append(aq_rest, ignore_index=True, verify_integrity=True)\
            .append(aq_live, ignore_index=True, verify_integrity=True)

        # Remove reports with no station_id
        aq.dropna(subset=[const.ID], inplace=True)

        # Drop possible duplicates
        aq.drop_duplicates(subset=[const.ID, const.TIME], inplace=True)

        # Convert datetime strings to objects
        aq[const.TIME] = pd.to_datetime(aq[const.TIME], utc=True)

        # Change negative reports of pollutants to NaN
        pollutants = [const.PM25, const.PM10]
        for pollutant in pollutants:
            negatives = aq[pollutant] < 0
            aq.loc[negatives, pollutant] = np.nan
            print('Negative {p} reports: {c}'.format(p=pollutant, c=np.sum(negatives)))

        # Re-arrange columns order for better readability
        self.obs = aq[[const.ID, const.TIME, const.PM25, const.PM10, 'NO2']]

        # Build and clean station data
        aq_stations.drop(columns=['api_data', 'historical_data', 'SiteName'], inplace=True)
        aq_stations.rename(columns={'Unnamed: 0': const.ID, 'need_prediction': const.PREDICT
            ,'Latitude': const.LAT, 'Longitude': const.LONG, 'SiteType': const.S_TYPE}, inplace=True)
        aq_stations[const.S_TYPE] = aq_stations[const.S_TYPE].str.lower()\
            .replace('urban background', 'urban')
        # Stations that has type are air quality ones and need to be predicted
        aq_stations[const.PREDICT] = (1 - aq_stations[const.PREDICT].isna()).astype(int)
        self.stations = aq_stations

        # Sort data
        self.sort()

        return self


if __name__ == "__main__":
    pre_process = PreProcessLD(settings.config[const.DEFAULT])
    pre_process.process()
    # pre_process.fill()
    print('No. observed rows:', len(pre_process.obs))
    print('No. stations:', len(pre_process.stations))
    pre_process.save()
    print("Done!")
