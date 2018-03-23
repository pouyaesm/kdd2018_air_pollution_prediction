import settings, const
import pandas as pd
import numpy as np

# access default configurations
config = settings.config[const.DEFAULT]

# read csv files
qualityDf = pd.read_csv(config[const.AIR_QUALITY], low_memory=False)
weatherDf = pd.read_csv(config[const.WEATHER], low_memory=False)

# rename station_id to stationId in weather table (the same as air quality table)
weatherDf.rename(columns={'station_id': const.STATION_ID}, inplace=True)

# remove _aq, _meo postfixes from station ids
qualityDf[const.STATION_ID] = qualityDf[const.STATION_ID].str.replace('_.*', '')  # name_aq -> name
weatherDf[const.STATION_ID] = weatherDf[const.STATION_ID].str.replace('_.*', '')  # name_meo -> name

# create an integer timestamp column from datetime for ease of processing
qualityDf['timestamp'] = np.divide(pd.to_datetime(qualityDf.utc_time).values.astype(np.int64), 1000000000)
weatherDf['timestamp'] = np.divide(pd.to_datetime(weatherDf.utc_time).values.astype(np.int64), 1000000000)

# ---- change invalid values to NaN -----
weatherDf.loc[weatherDf['temperature'] > 100] = np.nan  # max value ~ 40 c
weatherDf.loc[weatherDf['wind_speed'] > 100] = np.nan  # max value ~ 15 m/s
weatherDf.loc[weatherDf['wind_direction'] > 360] = np.nan  # value = [0, 360] degree
weatherDf.loc[weatherDf['humidity'] > 100] = np.nan  # value = [0, 100] percent

# Merge air quality and weather data based on stationId-timestamp
indices = ['stationId', 'utc_time']
df = qualityDf.merge(weatherDf, how='outer', left_on=indices, right_on=indices,
                     suffixes=['_aq', '_meo'])
# only keep the maximum of timestamp_aq and time_stamp_meo (that is ignore the NaN one)
df["timestamp"] = df[["timestamp_aq", "timestamp_meo"]].max(axis=1)
del df["timestamp_aq"]
del df["timestamp_meo"]

# print parts of table
print('No. merged rows:', len(df.index))

# Write cleaned data to csv file
df.to_csv(config['cleanData'], sep=';', index=False)
print('Cleaned data saved.')
# note: if file not opened properly in excel,
# go to Control panel > Region and language > additional settings
# and change the 'List separator' to ';'
