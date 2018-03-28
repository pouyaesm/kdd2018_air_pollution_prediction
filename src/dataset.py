import settings, const
from src import util
import pandas as pd
import datetime as dt


def get_pollutants_per_station(from_time='0000-00-00 00:00:00', to_time='9999-01-01 00:00:00'):
    # access default configurations
    config = settings.config[const.DEFAULT]

    # read csv files
    data = pd.read_csv(config[const.CLEAN_DATA], sep=";", low_memory=False)

    # discard report of stations that do not report air quality
    data = data.loc[data['aq'] == 1, :]

    # filter a specific time interval
    data['utc_time'] = pd.to_datetime(data['utc_time'])
    data = util.filter_by_time(data, time_key='utc_time', from_time=from_time, to_time=to_time)

    # list of stations to iterate and predict
    stations = data.drop_duplicates(subset=[const.STATION_ID])[const.STATION_ID].tolist()

    # put each station in a separate data-frame
    pollutants = {}
    for station in stations:
        # extract pollutants of specific station
        pollutants[station] = data.loc[data[const.STATION_ID] == station
        , ['utc_time', 'PM2.5', 'PM10', 'O3']].reset_index(drop=True)
        # fill missing values with nearest neighbors (same column)
        util.fill_missing(pollutants[station]['PM2.5'], inplace=True)
        util.fill_missing(pollutants[station]['PM10'], inplace=True)
        util.fill_missing(pollutants[station]['O3'], inplace=True)
    return pollutants
