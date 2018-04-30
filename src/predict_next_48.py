import pandas as pd
from datetime import datetime
from datetime import timedelta
import settings
import const
from src.preprocess.preprocess import PreProcess
from src.preprocess import times
from src.feature_generators.hybrid_fg import HybridFG
from src.preprocess import reform
from src.methods.hybrid import Hybrid

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

model_cfgs = {
    'BJ': {
        const.CITY: const.BJ,
        const.MODEL_DIR: config[const.MODEL_DIR],
        const.POLLUTANT: None,
        const.LOSS_FUNCTION: const.MEAN_PERCENT,
        const.CHUNK_COUNT: 10,
        const.TIME_STEPS: 12
    },
    'LD': {
        const.CITY: const.LD,
        const.MODEL_DIR: config[const.MODEL_DIR],
        const.POLLUTANT: None,
        const.LOSS_FUNCTION: const.MEAN_PERCENT,
        const.CHUNK_COUNT: 4,
        const.TIME_STEPS: 12
    }
}
today = times.to_datetime(datetime.utcnow().date())
tomorrow = times.to_datetime(today + timedelta(days=1))
now = datetime.utcnow()

for city, pollutants in cases.items():
    observed = pd.read_csv(config[getattr(const, city + "_OBSERVED")], sep=';', low_memory=True)
    stations = pd.read_csv(config[getattr(const, city + "_STATIONS")], sep=';', low_memory=True)
    observed = times.select(df=observed, time_key=const.TIME, from_time='18-04-15 00')
    observed[const.TIME] = pd.to_datetime(observed[const.TIME], format=const.T_FORMAT)
    model_cfg = model_cfgs[city]
    # # Fill all remaining null values that were to wide to be filled in general pre-processing
    # nan_rows = pd.isnull(observed[const.PM25]).sum()
    # pre_process = PreProcess().fill(observed=observed, stations=stations)
    # observed = pre_process.get_observed()
    # print('Nan PM2.5 before {b} and after {a} filling'.format(
    #     b=nan_rows, a=pd.isnull(observed[const.PM25]).sum()))

    features = dict()
    station_features = dict()

    for pollutant in pollutants:
        # generate features to predict the pollutant on next 48 hours
        pollutant_feature = HybridFG(time_steps=0, cfg={const.POLLUTANT: pollutant}).generate(
            ts=observed, stations=stations, verbose=False, save=False).get_features()

        # only keep features with time = 23:00
        pollutant_feature = times.select(pollutant_feature, const.TIME,
                                         from_time=tomorrow - timedelta(hours=1), to_time=tomorrow)

        # explode features to model inputs
        context, meo, future, air, label = HybridFG(cfg={}, time_steps=12).explode(
            features=features[pollutant])

        # initial model to predict
        model_cfg[const.POLLUTANT] = pollutant
        model = Hybrid(model_cfg).load_model(mode='best')
        output = model.predict(x={'c': context, 'm': meo, 'f': future, 'a': air, 'l': label})

        print(city, pollutant, 'done.')

def append(features, output, pollutant):
    return 1

