import numpy as np
from numpy import exp
import pandas as pd
import math
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


def split_dual(time: pd.Series, value: pd.Series, unit_x: int, unit_y: int):
    """
        Split data into (time, x, y) tuples
        This function is able to construct x: (past 24h, past 7d), y: (next 24h)
        for values: (hour, day), unit_x: (24, 7), unit_y: (24, 0)
        Note: 'h' category must exist by default
    :param time: time series of dictionaries
    :param value: time series of values
    :param unit_x: number of units per x split
    :param unit_y: number of units per y split
    :return:
    """
    x = list()  # values of first "unit_x"
    y = list()  # values of next "unit_y"
    # number of hours determines the overall split count
    split_count = len(value) - unit_x - unit_y + 1
    t = time[0:split_count]
    for i in range(0, split_count):
        split_at = i + unit_x
        # first "hours" values
        x.append(value[i:split_at])
        # next "hours" values as output
        y.append(value[split_at:split_at + unit_y])
    return t, x, y


def split(time: list, value: list, step, shift=1, skip=0):
    """
        Split data by unit into (time, value[0:unit]) tuples
    :param time: series of datetime elements
    :param value: series of values
    :param step: number of values extracted for each split
    :param shift: number of shift in units to extract the next split
    :param skip: number of units skipped at the first of list
    :return:
    """
    # effective length to utilize = K * step + unit + offset
    length = len(value) - step - skip  # when step = 1
    split_count = length - length % shift
    x = list()
    t = time[skip:split_count + skip]
    for i in range(0, split_count):
        x.append(value[i * shift + skip:i * shift + skip + step])
    return t, x


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


def average(value: list, step: int):
    """
        Put average of step values at each index, by averaging values behind
        or ahead of that index depending on 'step' sign
    :param value:  time series of values to be grouped
    :param step: number of steps to aggregate (positive, forward, negative backward)
    :return:
    """
    iteration = range(0, len(value)) if np.sign(step) > 0 else reversed(range(0, len(value)))
    step = abs(step)
    size = int(math.ceil(len(value) / (step + 1)))
    avg = [0] * len(value)  # average of values at index i corresponding to left or right neighbors of i
    aggregate = [0] * size  # sum of values for a time group
    count = [0] * size  # number of values for a time group

    for i in iteration:
        round_i = i // (step + 1)
        aggregate[round_i] = aggregate[round_i] + value[i]
        count[round_i] = count[round_i] + 1
        avg[i] = aggregate[round_i] / count[round_i]

    return avg


def wind_transform(speed: pd.Series, direction: pd.Series):
    """
        0 is from north to south -> 90
        90 is from east to west -> 180
    :param speed:
    :param direction:
    :return:
    """
    def to_polar(r, teta):
        x = np.multiply(r, np.cos(teta))
        y = np.multiply(r, np.sin(teta))
        return x, y

    teta = np.multiply(2 * math.pi, np.divide(np.mod(450 - direction, 360), 360.))
    x, y = to_polar(speed, teta)
    return x, y
