"""
    Predict each pollutant for each station using the simplest baseline, that is:
        f(t, d + 2) = f(t, d + 1) = a(t, d)
"""
import numpy as np
import pandas as pd
import const, settings
from src import util
from src.preprocess import reform, times

# access default configurations
config = settings.config[const.DEFAULT]

# load data
stations = pd.read_csv(config[const.BJ_STATIONS], sep=";", low_memory=False)
data = pd.read_csv(config[const.BJ_OBSERVED], sep=";", low_memory=False)
data = times.select(df=data, time_key=const.TIME,
                    from_time='18-01-01 00', to_time='18-01-31 23')
data_grouped = reform.group_by_station(ts=data, stations=stations)

pollutants = ['PM2.5']  #['PM2.5', 'PM10', 'O3']
columns = ['forecast', 'actual', 'station', 'pollutant']
predictions = pd.DataFrame(data={}, columns=columns)
for station in data_grouped:
    station_data = data_grouped[station]
    station_time = pd.to_datetime(station_data[const.TIME], format=const.T_FORMAT, utc=True)
    for pollutant in pollutants:
        t, x, y = reform.split_dual(
            time=station_time, value=station_data[pollutant], unit_x=24, unit_y=48)
        # use day d values as forecast of days d+1 and d+2
        x = np.array(x)
        y = np.array(y)
        double_x = np.concatenate((x, x), axis=1)
        forecast = double_x.reshape(double_x.size)
        actual = y.reshape(y.size)
        df = pd.DataFrame(data={'forecast': forecast, 'actual': actual}, columns=columns)
        df['station'] = station
        df['pollutant'] = pollutant
        predictions = predictions.append(df, ignore_index=True)

# drop NaN records
predictions.dropna(inplace=True)

print('SMAPE', util.SMAPE(forecast=predictions['forecast'], actual=predictions['actual']))




