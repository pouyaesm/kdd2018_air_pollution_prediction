import numpy as np
import pandas as pd


# Groups data into specific time intervals, then returns aggregated values of
# those with f(time) = value; both intervals and matching rows depend on the given mode
# modes:
#   h: hour of day [0, 23]
#   dw: day of week [0, 6], where 0: sunday, 6: saturday
#   w: week of month [0, 3]
#   m: month of year [1, 12]
#   s: season of year [1, 3]
# dt.<fields>: https://pandas.pydata.org/pandas-docs/stable/api.html#datetimelike-properties
def sample_time(time_series: pd.DataFrame, mode, value=None, time_key='time', agg_op=None):
    agg_op = {'value': 'mean'} if agg_op is None else agg_op
    sample = time_series.copy()
    datetime = pd.to_datetime(time_series[time_key])
    if mode == 'h':
        sample[time_key] = datetime.dt.strftime("%Y-%m-%d %H:00:00")  # coarse-grain the time to hour of day
        matches_value = datetime.dt.hour == value
    if mode == 'dw':
        sample[time_key] = datetime.dt.strftime("%Y-%W-%w")  # coarse-grain the time to day of week
        # in pandas dayofweek is 0: monday, 6: sunday,
        # unlike strftime where 0: sunday, 6:saturday
        matches_value = datetime.dt.dayofweek == ((value + 6) % 7)
    elif mode == 'm':
            sample[time_key] = datetime.dt.strftime("%Y-%m")  # coarse-grain the time to month
            matches_value = datetime.dt.month == value
    elif mode == 's':
        sample[time_key] = np.mod(datetime.dt.strftime('%Y-%m').astype(int), 4)

    # sample those with matching time values
    sample = sample.loc[matches_value, :] if value is not None else sample
    # aggregate values based on mode
    sample = sample.groupby([time_key], as_index=False).agg(agg_op)
    return sample.reset_index(drop=True)


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
