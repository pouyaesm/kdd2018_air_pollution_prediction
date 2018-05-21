import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import settings
import const
from src import util
from src.preprocess import times
from src.preprocess.preprocess import PreProcess
from src.feature_generators.hybrid_fg import HybridFG
from src.methods.hybrid import Hybrid


def predict(date_borders=None):
    today = times.to_datetime(datetime.utcnow().date())
    if date_borders is None:
        date_borders = [times.to_datetime(today + timedelta(days=1))]  # tomorrow
    batch_folder = "batch\\" if len(date_borders) > 1 else ""
    outputs = {d_border: list() for d_border in date_borders}
    actuals = {d_border: list() for d_border in date_borders}
    smape_total = {d_border: {const.BJ: 0, const.LD: 0} for d_border in date_borders}
    smape_count = {d_border: {const.BJ: 0, const.LD: 0} for d_border in date_borders}
    for city, pollutants in cases.items():
        observed = pd.read_csv(config[getattr(const, city + "_OBSERVED")], sep=';', low_memory=True)
        stations = pd.read_csv(config[getattr(const, city + "_STATIONS")], sep=';', low_memory=True)
        # keep only a necessary time range for feature generation for only predictable stations
        observed = times.select(df=observed, time_key=const.TIME, from_time='18-04-20 00')
        observed[const.TIME] = pd.to_datetime(observed[const.TIME], format=const.T_FORMAT)
        observed.sort_values(by=[const.ID, const.TIME], inplace=True)
        # Fill all remaining null values that were to wide to be filled in general pre-processing
        # nan_rows = pd.isnull(observed[const.PM25]).sum()
        pre_process = PreProcess().fill(observed=observed, stations=stations)
        observed = pre_process.get_observed()
        # print('Nan PM2.5 before {b} and after {a} filling'.format(
        #     b=nan_rows, a=pd.isnull(observed[const.PM25]).sum()))

        # Model configuration (city dependant)
        model_cfg = model_cfgs[city]
        model_cfg.update(model_basic_cfg)
        fg_cfg_basic = HybridFG.get_size_config(city=city, key=feature_mode)
        fg_cfg_basic.update({const.GRID_COARSE: model_cfg[const.GRID_COARSE]})
        model_cfg.update(fg_cfg_basic)
        model_cfg[const.STATIONS] = config[getattr(const, city + '_STATIONS')]

        for pollutant in pollutants:
            print(city, pollutant)
            fg_cfg = {const.POLLUTANT: pollutant}
            fg_cfg.update(fg_cfg_basic)
            fg = HybridFG(cfg=fg_cfg)
            # initial model to predict
            model_cfg[const.POLLUTANT] = pollutant
            model = Hybrid(model_cfg).load_model(mode='best')
            # generate features to predict the pollutant on next 48 hours
            all_features = fg.generate(ts=observed, stations=stations, verbose=False, save=False).get_features()
            for d_border in date_borders:
                date_start = d_border - timedelta(hours=1)
                # only keep features with time = 23:00
                features = times.select(all_features, const.TIME,
                                        from_time=d_border - timedelta(hours=1),
                                        to_time=d_border)

                # explode features to model inputs
                exploded = fg.explode(features=features)
                predictions = model.predict(x=exploded)
                # add _aq prefix back for beijing stations
                station_col = (features[const.ID] + '_aq') if city == const.BJ else features[const.ID]
                pollutant_col = [pollutant for _ in range(0, len(predictions))]
                output = [[sid] + [pol] + p for sid, pol, p in
                          zip(station_col.tolist(), pollutant_col, predictions.tolist())]
                # add (city, pollutant) result to all outputs
                outputs[d_border].extend(output)
                # test the model if date_border is set before today
                if d_border <= today:
                    actual = features[fg.get_label_columns()].as_matrix()
                    smape = util.SMAPE(forecast=predictions, actual=actual)
                    smape_total[d_border][city] += smape * actual.size
                    smape_count[d_border][city] += actual.size
                    print(' %s %s %s, SMAPE %.3f' % (
                        city, pollutant, date_start, smape))
                    # keep actual values for further analysis
                    actual = [[sid] + [pol] + p for sid, pol, p in
                              zip(station_col.tolist(), pollutant_col, actual.tolist())]
                    actuals[d_border].extend(actual)
                else:
                    print(' %s %s %s' % (city, pollutant, date_start))
            print('_________________________')

    for d_border in date_borders:
        # convert result to desired format (station_id, hour, pm2.5, pm10, o3)
        results_predict = list()
        results_actual = list()
        mapper = dict()
        row = 0
        pollutant_position = {const.PM25: 1, const.PM10: 2, const.O3: 3}
        for i, output in enumerate(outputs[d_border]):
            sid = output[0]
            pollutant = output[1]
            actual = actuals[d_border][i][2:] if d_border <= today else output[2:]
            for hour, value in enumerate(output[2:]):
                row_key = '%s#%s' % (sid, hour)
                if row_key not in mapper:
                    mapper[row_key] = row
                    results_predict.append([row_key, np.nan, np.nan, np.nan])
                    results_actual.append([row_key, np.nan, np.nan, np.nan])
                    row += 1
                results_predict[mapper[row_key]][pollutant_position[pollutant]] = value
                results_actual[mapper[row_key]][pollutant_position[pollutant]] = actual[hour]

        if d_border <= today:
            bj_smape = smape_total[d_border][const.BJ] / smape_count[d_border][const.BJ]
            ld_smape = smape_total[d_border][const.LD] / smape_count[d_border][const.LD]
            smape = (bj_smape + ld_smape) / 2
            print('%s %.3f' % (d_border, smape))
            df = pd.DataFrame(data=results_actual, columns=['test_id', 'PM2.5', 'PM10', 'O3'])
            actual_path = config[const.SUBMIT_DIR] + batch_folder \
                          + "result___" + \
                          (d_border - timedelta(hours=1)).strftime("%Y_%m_%d") + "_actual.csv"
            df.to_csv(actual_path, sep=',', index=False, float_format='%.3f')

        df = pd.DataFrame(data=results_predict, columns=['test_id', 'PM2.5', 'PM10', 'O3'])
        submit_path = config[const.SUBMIT_DIR] + batch_folder \
                      + "result___" + \
                      (d_border - timedelta(hours=1)).strftime("%Y_%m_%d") + ".csv"
        df.to_csv(submit_path, sep=',', index=False, float_format='%.3f')


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]

    today = times.to_datetime(datetime.utcnow().date())
    # tomorrow
    date_borders = None
    # 2 days before
    date_borders = [times.to_datetime(today - timedelta(days=day)) for day in range(19, 1, -1)]

    feature_mode = '05-02'
    model_class = Hybrid
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
        const.MODEL_DIR: config[const.MODEL_DIR] + 'production\\',
        const.STATIONS: None,
        const.POLLUTANT: None,
        const.LOSS_FUNCTION: const.MEAN_PERCENT,
    }

    model_cfgs = {
        const.BJ: {
            const.GRID_COARSE: config[getattr(const, 'BJ_GRID_COARSE')],
        },
        const.LD: {
            const.GRID_COARSE: config[getattr(const, 'LD_GRID_COARSE')],
        }
    }

    predict(date_borders)
