import settings
import const
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})  # to prevent labels going out of plot!
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
from src.feature_generators.lstm_fg import LSTMFG
from src.preprocess import reform
from src import util


config = settings.config[const.DEFAULT]
suffix = '48_lstm_test.csv'
paths = {
    'BJ': {
        'PM25': config[const.BJ_PM25_] + suffix,
        # 'PM10': config[const.BJ_PM10_] + suffix,
        # 'O3': config[const.BJ_O3_] + suffix,
    },
    # 'LD': {
    #     'PM25': config[const.LD_PM25_] + suffix,
    #     'PM10': config[const.LD_PM10_] + suffix,
    # }
}

# fig_bj_pm25, axes_bj_pm25 = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))
# fig_bj_pm10, axes_bj_pm10 = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))
# fig_bj_o3, axes_bj_o3 = plt.subplots(nrows=1, ncols=2, figsize=(15, 3))
# fig_ld_pm25, axes_ld_pm25 = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
# fig_ld_pm10, axes_ld_pm10 = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
# axes = {
#     'plot':{'BJ': {'PM25': axes_bj_pm25[0], 'PM10': axes_bj_pm10[0], 'O3': axes_bj_pm10[3]}
#             , 'LD': {'PM25': axes_ld[1][0], 'PM10': axes_ld[1][1]}}
#     'scatter':{'BJ': {'PM25': axes_bj_pm25[0], 'PM10': axes_bj[0][1], 'O3': axes_bj[0][2]}
#     , 'LD': {'PM25': axes_ld[0][0], 'PM10': axes_ld[0][1]}},
# }


smape_columns = ['city', const.ID, const.LONG, const.LAT, 'pollutant', 'SMAPE']
for city in paths:
    station_path = config[const.BJ_STATIONS] if city == 'BJ' else config[const.LD_STATIONS]
    stations = pd.read_csv(station_path, sep=";", low_memory=False)
    for pollutant, path in paths[city].items():
        ts = pd.read_csv(path, sep=";", low_memory=False)
        station_data = reform.group_by_station(ts=ts, stations=stations)
        stations = stations.to_dict(orient='index')
        smapes = pd.DataFrame(data=[], columns=smape_columns)
        for _, station in stations.items():
            data = station_data[station[const.ID]] if station[const.PREDICT] == 1 else pd.DataFrame()
            if len(data.index) == 0:
                continue  # no prediction for this station
            actual = data[['l' + str(i) for i in range(0, 48)]].as_matrix()
            forecast = data[['f' + str(i) for i in range(0, 48)]].as_matrix()
            station['SMAPE'] = util.SMAPE(actual=actual, forecast=forecast)
            smape = pd.DataFrame(
                    data=[[city, station[const.ID], station[const.LONG], station[const.LAT],
                           pollutant, station['SMAPE']]],
                    columns=smape_columns)
            smapes = smapes.append(other=smape, ignore_index=True)
            # print(city, ', ', station, ', ', pollutant, ' ', smape_score)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 3))

        # Plot SMAPE values sorted
        smapes.sort_values(by='SMAPE', inplace=True)
        g = sns.stripplot(x=const.ID, y='SMAPE', data=smapes, ax=axes[0])
        g.set_xticklabels(labels=g.get_xticklabels(), rotation=90)  # rotate station names for readability

        # Plot SMAPE values on map
        smapes.plot.scatter(x=const.LONG, y=const.LAT, s=util.normalize(smapes['SMAPE'], multiplier=150),
                            title=city + '_' + pollutant, fontsize=13, ax=axes[1])
        # Plot station names on positions
        for _, station in stations.items():
            if 'SMAPE' in station:
                label = ('%d ' % (100 * station['SMAPE'])) + station[const.ID][0:2]  # 64 be
                axes[1].annotate(label, xy=(station[const.LONG], station[const.LAT]),
                                 xytext=(5, 0), textcoords='offset points', )

plt.show()
