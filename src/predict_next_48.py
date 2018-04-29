import pandas as pd
from datetime import datetime
from datetime import timedelta
import settings
import const
from src.preprocess.preprocess import PreProcess
from src.preprocess import times
from src.feature_generators.hybrid_fg import HybridFG
from src.preprocess import reform

config = settings.config[const.DEFAULT]

cases = {
    'BJ': {
        'PM2.5': 200000,
        'PM10': -1,
        'O3': 200000,
    },
    'LD': {
        'PM2.5': -1,
        'PM10': -1,
    }
}

today = times.to_datetime(datetime.utcnow().date())
tomorrow = times.to_datetime(today + timedelta(days=1))

for city, pollutants in cases.items():
    observed = pd.read_csv(config[getattr(const, city + "_OBSERVED")], sep=';', low_memory=True)
    stations = pd.read_csv(config[getattr(const, city + "_STATIONS")], sep=';', low_memory=True)
    observed = times.select(df=observed, time_key=const.TIME, from_time='18-04-01 00')
    observed[const.TIME] = pd.to_datetime(observed[const.TIME], format=const.T_FORMAT)

    # Fill all remaining null values that were to wide to be filled in general pre-processing
    nan_rows = pd.isnull(observed[const.PM25]).sum()
    pre_process = PreProcess().fill(observed=observed, stations=stations)
    observed = pre_process.get_observed()
    print('Nan PM2.5 before {b} and after {a} filling'.format(
        b=nan_rows, a=pd.isnull(observed[const.PM25]).sum()))

    all_features = dict()
    station_features = dict()
    for pollutant in pollutants:
        features = HybridFG(time_steps=0, cfg={const.POLLUTANT: pollutant}).generate(
            ts=observed, stations=stations, verbose=False, save=False).get_features()
        station_features[pollutant] = reform.group_by_station(ts=features, stations=stations)

    for pollutant in pollutants:
        # station_features = reform.group_by_station(ts=all_features, stations=stations)
        for station_id, features in station_features.items():
            last_feature = features.ix[features[const.TIME].idxmax()]
            time_lag = tomorrow - last_feature[const.TIME].to_pydatetime()
            # predict to cover the time_lag to 00:00
            a = 1
            # predict for next 48 hours

        print(city, pollutant, 'done.')

def load_model(city, pollutant):
    return 1