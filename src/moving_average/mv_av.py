from src import util
from src import dataset
import pandas as pd
import numpy as np

data = dataset.get_pollutants_per_station(from_time='2017-12-31', to_time='2018-01-31')
stations = data.keys()
pollutants = ['PM2.5', 'PM10', 'O3']

# predict each pollutant for each station using f(t) = xt-1 (simplest baseline)
index = 0
columns = ['forecast', 'actual', 'station', 'pollutant']
predictions = pd.DataFrame(data={}, columns=columns)
for pollutant in pollutants:
    for station in stations:
        time_series = data[station][pollutant]
        # extract (xt-1, xt) blocks as (x, y),
        # xt-1 (x) is considered as the estimation of real value xt (y)
        x, y = util.window_for_predict(time_series, 1, 1, 1)
        forecast = np.array(x).reshape(len(x))
        actual = np.array(y).reshape(len(y))
        df = pd.DataFrame(data={'forecast': forecast, 'actual': actual}, columns=columns)
        df['station'] = station
        df['pollutant'] = pollutant
        predictions = predictions.append(df, ignore_index=True)

# drop NaN records
predictions.dropna(inplace=True)

print('SMAPE', util.SMAPE(forecast=predictions['forecast'], actual=predictions['actual']))




