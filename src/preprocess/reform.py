import numpy as np
import pandas as pd
import const


def group_by_station(ts: pd.DataFrame, stations: pd.DataFrame):
    """
        Group data series by station
    :param ts: time series data
    :param stations: station info
    :return:
    """
    grouped = {}
    station_ids = stations.loc[stations[const.PREDICT] == 1, const.ID]
    for _, station_id in enumerate(station_ids):
        # extract pollutants of specific station
        grouped[station_id] = ts.loc[ts[const.ID] == station_id, :].reset_index(drop=True)
    return grouped


def window_for_predict(values: pd.Series, x_size, y_size, step):
    """
    converts a time-series into windowed (x, y) samples
    values=[1, 2, 3, 4], x_size = 2, y_size = 1, step = 1
    creates x=[[1, 2],[2, 3]], y=[[3],[4]]

    :param values:
    :param x_size: no. of time steps as input of prediction
    :param y_size: no. of time steps as output of prediction
    :param step:
    :return: windowed input, output
    """
    last_input = values.size - y_size - 1  # last input right before last output
    first_output = x_size  # index of first output right after first input
    window_x = window(values.loc[0:last_input], x_size, step)
    window_y = window(values.loc[first_output:values.size - 1].reset_index(drop=True), y_size, step)
    return window_x, window_y


def split_by_hours(time: list, value: list, hours_x, hours_y):
    """
        Split data by hours into (time, input, output) tuples
        Data is assumed to be sorted by time incrementally
    :param time: series of datetime elements
    :param value: series of values
    :param miss: series of missing indicators (0 and 1)
    :param hours_x: number of hours per x split
    :param hours_y: number of hours per y split
    :return:
    """
    split_count = len(value) - hours_x - hours_y + 1
    x = list()  # values of first "hours"
    y = list()  # values of next "hours"
    t = time[0:split_count]
    for i in range(0, split_count):
        # first "hours" values
        x.append(value[i:i + hours_x])
        # next "hours" values as output
        y.append(value[i + hours_x:i + hours_x + hours_y])
    return t, x, y


def window(values: pd.Series, window_size, step):
    """
    converts a series into overlapping blocks
    for example: ts = [1, 2, 3, 4, 5, 6], window_size = 3, step = 2, skip = 0
    creates [[1, 2, 3], [3, 4, 5]]

    :param values:
    :param window_size:
    :param step: move step times to extract the next window
    :return:
    """
    # values length must be = k * step + window_size, for some k
    # so we trim the reminder to reach this equation
    reminder = (values.size - window_size) % step
    trimmed = values.loc[0:(values.size - 1 - reminder)]
    shape = trimmed.shape[:-1] + (int((trimmed.shape[-1] - window_size) / step + 1), window_size)
    strides = (step * trimmed.strides[-1],) + (trimmed.strides[-1],)
    windowed = np.lib.stride_tricks.as_strided(trimmed, shape=shape, strides=strides)
    return windowed
