"""
    predict each pollutant for each station using the simplest baseline, that is:
        f(t, d + 2) = f(t, d + 1) = a(t, d)
"""

from src import util
from src.dataset import DataSet
import pandas as pd
import numpy as np

data_set = DataSet().load(from_time='2017-12-31', to_time='2018-01-31')

pollutants = ['PM2.5', 'PM10', 'O3']
columns = ['forecast', 'actual', 'station', 'pollutant']
predictions = pd.DataFrame(data={}, columns=columns)

for pollutant in pollutants:
    pollutant_data = data_set.get_pollutant(pollutant)
    for station in data_set.stations:
        is_nan = pollutant_data[station]['is_nan']
        time_series = pollutant_data[station][pollutant]
        # extract (xt-1, xt) blocks as (x, y),
        # xt-1 (x) is considered as the estimation of real value xt (y)
        x, y = util.window_for_predict(time_series, 2, 4, 2)
        forecast = np.array(x).reshape(len(x))
        actual = np.array(y).reshape(len(y))
        df = pd.DataFrame(data={'forecast': forecast, 'actual': actual}, columns=columns)
        df['station'] = station
        df['pollutant'] = pollutant
        predictions = predictions.append(df, ignore_index=True)

# drop NaN records
predictions.dropna(inplace=True)

print('SMAPE', util.SMAPE(forecast=predictions['forecast'], actual=predictions['actual']))




