import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import settings
import const
from src import util
from src.preprocess import times
from src.feature_generators.hybrid_fg import HybridFG
from src.methods.hybrid import Hybrid

config = settings.config[const.DEFAULT]

cases = {
    'BJ': {
        'PM2.5': True,
        'PM10': True,
        'O3': True,
    },
    'LD': {
        'PM2.5': True,
        'PM10': True,
    }
}

model_basic_cfg = {
    const.MODEL_DIR: config[const.MODEL_DIR],
    const.FEATURE: None,
    const.STATIONS: None,
    const.POLLUTANT: None,
    const.LOSS_FUNCTION: const.MEAN_PERCENT,
    const.CHUNK_COUNT: 10,
    const.TIME_STEPS: 12
}

model_cfgs = {
    'BJ': {
        const.CITY: const.BJ,
    },
    'LD': {
        const.CITY: const.LD,
    }
}
today = times.to_datetime(datetime.utcnow().date())
# tomorrow
date_border = times.to_datetime(today + timedelta(days=1))
# # 2 days before
# date_border = times.to_datetime(today - timedelta(days=2))
now = datetime.utcnow()

print('Date border:', date_border.strftime(const.T_FORMAT_FULL))

outputs = list()
for city, pollutants in cases.items():
    observed = pd.read_csv(config[getattr(const, city + "_OBSERVED")], sep=';', low_memory=True)
    stations = pd.read_csv(config[getattr(const, city + "_STATIONS")], sep=';', low_memory=True)
    # keep only a necessary time range for feature generation for only predictable stations
    observed = times.select(df=observed, time_key=const.TIME, from_time='18-04-01 00')
    observed[const.TIME] = pd.to_datetime(observed[const.TIME], format=const.T_FORMAT)

    # # Fill all remaining null values that were to wide to be filled in general pre-processing
    # nan_rows = pd.isnull(observed[const.PM25]).sum()
    # pre_process = PreProcess().fill(observed=observed, stations=stations)
    # observed = pre_process.get_observed()
    # print('Nan PM2.5 before {b} and after {a} filling'.format(
    #     b=nan_rows, a=pd.isnull(observed[const.PM25]).sum()))

    # Model configuration (city dependant)
    model_cfg = model_cfgs[city]
    model_cfg.update(model_basic_cfg)
    model_cfg[const.STATIONS] = config[getattr(const, city + '_STATIONS')]

    for pollutant in pollutants:
        fg = HybridFG(cfg={const.CITY: city, const.POLLUTANT: pollutant})
        # generate features to predict the pollutant on next 48 hours
        features = fg.generate(ts=observed, stations=stations, verbose=False, save=False).get_features()

        # only keep features with time = 23:00
        features = times.select(features, const.TIME,
                                from_time=date_border - timedelta(hours=1), to_time=date_border)

        # explode features to model inputs
        context, meo, future, air, label = fg.explode(features=features)

        # initial model to predict
        model_cfg[const.POLLUTANT] = pollutant
        model_cfg[const.FEATURE] = getattr(const, city + '_' + pollutant.replace('.', '') + '_')
        model = Hybrid(model_cfg).load_model(mode='best')
        predictions = model.predict(x={'c': context, 'm': meo, 'f': future, 'a': air, 'l': label})
        # test the model if date_border is set before today
        if date_border < today:
            actual = features[fg.get_label_columns()].as_matrix()
            print(' Test SMAPE', util.SMAPE(forecast=predictions, actual=actual))
        # add _aq prefix back for beijing stations
        station_col = (features[const.ID] + '_aq') if city == const.BJ else features[const.ID]
        pollutant_col = [pollutant for _ in range(0, len(predictions))]
        output = [[sid] + [pol] + p for sid, pol, p in
                  zip(station_col.tolist(), pollutant_col, predictions.tolist())]
        # add (city, pollutant) result to all outputs
        outputs.extend(output)
        print(city, pollutant, 'predicted!')

# convert result to desired format (station_id, hour, pm2.5, pm10, o3)
results = list()
mapper = dict()
row = 0
pollutant_position = {const.PM25: 1, const.PM10: 2, const.O3: 3}
for i, output in enumerate(outputs):
    sid = output[0]
    pollutant = output[1]
    for hour, value in enumerate(output[2:]):
        row_key = '%s#%s' % (sid, hour)
        if row_key not in mapper:
            mapper[row_key] = row
            results.append([row_key, np.nan, np.nan, np.nan])
            row += 1
        results[mapper[row_key]][pollutant_position[pollutant]] = value

df = pd.DataFrame(data=results, columns=['test_id', 'PM2.5', 'PM10', 'O3'])
submit_path = config[const.SUBMIT_DIR] + "result_" + today.strftime("%Y_%m_%d") + ".csv"
df.to_csv(submit_path, sep=',', index=False, float_format='%.3f')
