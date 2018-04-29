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
from src.preprocess import reform
from src import util

config = settings.config[const.DEFAULT]
feature_dir = config[const.FEATURE_DIR]
suffix = '12_hybrid_tests.csv'
paths = {
    'BJ': {
        'PM2.5': feature_dir + const.BJ_PM25_ + suffix,
        # 'PM10': feature_dir + const.BJ_PM10_ + suffix,
        # 'O3': feature_dir + const.BJ_O3_ + suffix,
    },
    'LD': {
        # 'PM25': feature_dir + const.LD_PM25_ + suffix,
        # 'PM10': feature_dir + const.LD_PM10_ + suffix,
    }
}

smape_columns = ['city', const.ID, const.LONG, const.LAT, 'pollutant', 'SMAPE', 'count']
smapes = pd.DataFrame(columns=smape_columns)
for city in paths:
    station_path = config[const.BJ_STATIONS] if city == 'BJ' else config[const.LD_STATIONS]
    stations = pd.read_csv(station_path, sep=";", low_memory=False)
    stations_dict = stations.to_dict(orient='index')
    for pollutant, path in paths[city].items():
        ts = pd.read_csv(path, sep=";", low_memory=False)
        station_data = reform.group_by_station(ts=ts, stations=stations)
        local_smapes = pd.DataFrame(data=[], columns=smape_columns)
        for _, station in stations_dict.items():
            data = station_data[station[const.ID]] if station[const.PREDICT] == 1 else pd.DataFrame()
            if len(data.index) == 0:
                continue  # no prediction for this station
            actual = data[[pollutant + '__' + str(i) for i in range(1, 49)]].as_matrix()
            forecast = data[['f' + str(i) for i in range(0, 48)]].as_matrix()
            station['SMAPE'] = util.SMAPE(actual=actual, forecast=forecast)
            smape = pd.DataFrame(
                    data=[[city, station[const.ID], station[const.LONG], station[const.LAT],
                           pollutant, station['SMAPE'], actual.size]],
                    columns=smape_columns)
            local_smapes = local_smapes.append(other=smape, ignore_index=True)
            smapes = smapes.append(other=smape, ignore_index=True)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 3))

        # Plot SMAPE values sorted
        local_smapes.sort_values(by='SMAPE', inplace=True)
        g = sns.stripplot(x=const.ID, y='SMAPE', data=local_smapes, ax=axes[0])
        g.set_xticklabels(labels=g.get_xticklabels(), rotation=90)  # rotate station names for readability

        # Plot SMAPE values on map
        local_smapes.plot.scatter(x=const.LONG, y=const.LAT, s=util.normalize(local_smapes['SMAPE'], multiplier=150),
                            title=city + '_' + pollutant, fontsize=13, ax=axes[1])
        # Plot station names on positions
        for _, station in stations_dict.items():
            if 'SMAPE' in station:
                label = ('%d ' % (100 * station['SMAPE'])) + station[const.ID][0:2]  # 64 be
                axes[1].annotate(label, xy=(station[const.LONG], station[const.LAT]),
                                 xytext=(5, 0), textcoords='offset points', )
        plt.draw()

# Calculate total error
total_smape = np.sum(smapes['SMAPE'] * smapes['count']) / np.sum(smapes['count'])
print('Total SMAPE:', total_smape)

plt.show()