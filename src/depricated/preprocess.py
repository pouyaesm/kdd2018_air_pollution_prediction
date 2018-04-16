import settings, const
import pandas as pd
import numpy as np
from src import util

# access default configurations
config = settings.config[const.DEFAULT]

# read csv files
qualityDf = pd.read_csv(config[const.BJ_AQ], low_memory=False)
weatherDf = pd.read_csv(config[const.BJ_MEO], low_memory=False)
stationsDf = pd.read_csv(config[const.BJ_STATIONS], low_memory=False)

# rename stationId to ID in quality table (the same as air weather table)
qualityDf.rename(columns={'stationId': const.ID}, inplace=True)

# remove _aq, _meo postfixes from station ids
qualityDf[const.ID] = qualityDf[const.ID].str.replace('_.*', '')  # name_aq -> name
weatherDf[const.ID] = weatherDf[const.ID].str.replace('_.*', '')  # name_meo -> name

# create an integer timestamp column from datetime for ease of processing
qualityDf['timestamp'] = np.divide(pd.to_datetime(qualityDf.utc_time).values.astype(np.int64), 1000000000)
weatherDf['timestamp'] = np.divide(pd.to_datetime(weatherDf.utc_time).values.astype(np.int64), 1000000000)

# ---- change invalid values to NaN -----
weatherDf.loc[weatherDf['temperature'] > 100, 'temperature'] = np.nan  # max value ~ 40 c
weatherDf.loc[weatherDf['wind_speed'] > 100, 'wind_speed'] = np.nan  # max value ~ 15 m/s
weatherDf.loc[weatherDf['wind_direction'] > 360, 'wind_direction'] = np.nan  # value = [0, 360] degree
weatherDf.loc[weatherDf['humidity'] > 100, 'humidity'] = np.nan  # value = [0, 100] percent

# Merge air quality and weather data based on stationId-timestamp
indices = [const.ID, 'utc_time']
df = qualityDf.merge(weatherDf, how='outer', left_on=indices, right_on=indices,
                     suffixes=['_aq', '_meo'])
# only keep the max (non NaN) timestamp
df = util.merge_columns(df, main='_aq', auxiliary='_meo')

# Add air quality locations from stations file to the joined data set
# Locations in df are from _meo data set, and locations from stations file are for _aq stations
df = df.merge(stationsDf, how='outer', left_on=[const.ID], right_on=[const.ID],
              suffixes=['_meo', '_aq'])
# only keep the max (non NaN) location
df = util.merge_columns(df, main='_aq', auxiliary='_meo')

# remove rows without station id which their existence is weird!
df.dropna(subset=['ID'], inplace=True)

# create set of stations that are meteorological but not air quality
quality_stations = qualityDf.drop_duplicates(subset=[const.ID])[const.ID].tolist()
weather_stations = weatherDf.drop_duplicates(subset=[const.ID])[const.ID].tolist()
weather_only_stations = list(set(weather_stations) - set(quality_stations))

# mark stations that are air quality stations with 1 and others with 0
df['aq'] = 1 - df[const.ID].isin(weather_only_stations)

# print parts of table
print('No. merged rows:', len(df.index))
print('No. weather only stations:', len(weather_only_stations))

# Write cleaned data to csv file
df.to_csv(config['cleanData'], sep=';', index=False)
print('Cleaned data saved.')
# note: if file not opened properly in excel,
# go to Control panel > Region and language > additional settings
# and change the 'List separator' to ';'
