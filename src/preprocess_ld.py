import settings
import const
import pandas as pd
import numpy as np
from src import util


class PreProcessLD:

    def __init__(self, config):
        self.config = config  # location of input/output files
        self.obs = pd.DataFrame()  # merged observed air quality and meteorology data per station
        self.missing = pd.DataFrame()  # indicator of missing values in observed data-frame
        self.stations = pd.DataFrame()  # all stations with their attributes such as type and position

    def process(self):
        """
            Load and PreProcess the data
        :return:
        """
        # Read two parts of london observed air quality data
        aq = pd.read_csv(self.config[const.LD_AQ], low_memory=False)
        aq_rest = pd.read_csv(self.config[const.LD_AQ_REST], low_memory=False)

        # Remove the index column (first) of aq
        aq.drop(aq.columns[0], axis=1, inplace=True)

        # Rename columns to conventional ones, and change 'value (ug/m3)' to 'value'
        aq.rename(columns={'MeasurementDateGMT': const.TIME}, inplace=True)
        aq.rename(columns={col: col.split(' ')[0] for col in aq.columns}, inplace=True)
        aq_rest.rename(columns={'Station_ID': const.ID, 'MeasurementDateGMT': const.TIME}, inplace=True)
        aq_rest.rename(columns={col: col.split(' ')[0] for col in aq_rest.columns}, inplace=True)
        # Drop last two completely null columns from aq_rest
        aq_rest.dropna(axis=1, how='all', inplace=True)

        # Remove reports with no station_id
        aq.dropna(subset=[const.ID], inplace=True)
        aq_rest.dropna(subset=[const.ID], inplace=True)

        # Append aq_rest to aq
        aq = aq.append(aq_rest, ignore_index=True, verify_integrity=True)

        # Convert datetime strings to objects
        aq[const.TIME] = pd.to_datetime(aq[const.TIME], utc=True)

        # Re-arrange columns order for better readability
        aq = aq[[const.ID, const.TIME, 'PM2.5', 'PM10', 'NO2']]

        # Sort data first based on station ids (alphabetically), then by time ascending
        self.obs = aq.sort_values([const.ID, const.TIME], ascending=True)

        # mark missing values
        self.missing = self.obs.isna().astype(int)

        return self

    def save(self):
        """
            Save pre-processed data to files given in config
        :return:
        """
        # Write pre-processed data to csv file
        util.write(self.obs, self.config[const.LD_OBSERVED])
        util.write(self.missing, self.config[const.LD_OBSERVED_MISS])
        # self.missing.to_csv(, sep=';', index=False)
        self.stations.to_csv(self.config[const.LD_STATIONS], sep=';', index=False)
        print('Data saved.')


if __name__ == "__main__":
    pre_process = PreProcessLD(settings.config[const.DEFAULT])
    pre_process.process()
    print('No. observed rows:', len(pre_process.obs))
    # print('No. stations:', len(pre_process.stations),
    #       ', only weather:', pre_process.stations['station_type'].count())
    pre_process.save()
    print("Done!")
