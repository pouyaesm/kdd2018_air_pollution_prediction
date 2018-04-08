"""
    Predict each pollutant for each station using the simplest baseline, that is:
        f(t, d + 2) = f(t, d + 1) = a(t, d)
"""

import const, settings
from src import util
from src.preprocess import reform, time
import pandas as pd
import numpy as np

# access default configurations
config = settings.config[const.DEFAULT]

# load data
stations = pd.read_csv(config[const.BJ_STATIONS], sep=";", low_memory=False)
data = pd.read_csv(config[const.BJ_OBSERVED], sep=";", low_memory=False)
data = time.select(df=data, time_key=const.TIME,
                           from_time='18-01-01 00', to_time='18-01-31 23')
data_grouped = reform.group_by_station(ts=data, stations=stations)

pollutants = ['PM2.5']  #['PM2.5', 'PM10', 'O3']
columns = ['forecast', 'actual', 'station', 'pollutant']
predictions = pd.DataFrame(data={}, columns=columns)
for station in data_grouped:
    station_data = data_grouped[station]
    station_time = pd.to_datetime(station_data[const.TIME], format=const.T_FORMAT, utc=True)
    for pollutant in pollutants:
        x, y, time_x = reform.split_by_hours(
            time=station_time, value=station_data[pollutant], hours_x=24, hours_y=48)
        # use day d values as forecast of days d+1 and d+2
        trimmed_x = x[:, 2:]
        double_x = np.concatenate((trimmed_x, trimmed_x), axis=1)
        forecast = double_x.reshape(double_x.size)
        actual = y.reshape(y.size)
        df = pd.DataFrame(data={'forecast': forecast, 'actual': actual}, columns=columns)
        df['station'] = station
        df['pollutant'] = pollutant
        predictions = predictions.append(df, ignore_index=True)

# drop NaN records
predictions.dropna(inplace=True)

print('SMAPE', util.SMAPE(forecast=predictions['forecast'], actual=predictions['actual']))




