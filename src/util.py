# general utilities used throughout the project
import numpy as np
import pandas as pd


# convert time string to season
def to_season(time):
    datetime = pd.to_datetime(time)
    return (datetime.month % 12 + 3) // 3 if datetime is not np.nan else np.nan


# normalize values of data-frame to [0, 1]
def normalize(data_frame, multiplier):
    return multiplier * (data_frame - data_frame.min()) / (data_frame.max() - data_frame.min())


# convert float[s] to pretty
def pretty(value, decimal):
    if isinstance(value, list):
        return [("%0." + str(decimal) + "f") % y for y in value]
    else:
        return ("%0." + str(decimal) + "f") % value

def fill(series: pd.Series, inplace=False):
    """
        Replace NaN values with average of nearest non NaN neighbors
    :param series:
    :param inplace:
    :return:
    """
    filled = series if inplace else series.copy()
    region = [-1, -1]  # temporary interval of NaN values
    last_item = len(filled) - 1
    lbound = filled.index[0]  # index bounds of the series
    ubound = filled.index[last_item]
    for index, value in filled.iteritems():
        # Keep track of current interval of NaN values
        if np.isnan(value):
            if region[0] == -1:
                region[0] = index
            region[1] = index
        # Replace NaN values with their boundary average
        # when a NaN interval is started, and is ending with a non-NaN value or end of list
        if region[0] != -1 and (not np.isnan(value) or index == ubound):
            start = region[0] - lbound  # offset index to 0
            end = region[1] - lbound  # offset index to 0
            first_value = filled.values[start - 1] if region[0] > lbound else np.nan
            last_value = filled.values[end + 1] if region[1] < ubound else np.nan
            # Duplicate one boundary to another if one does not exist
            # this happens when a series starts or ends with a NaN
            first_value = last_value if np.isnan(first_value) else first_value
            last_value = first_value if np.isnan(last_value) else last_value
            # Set average of boundaries for the NaN interval
            filled.values[start:end + 1] = \
                [(first_value + last_value) / 2] * (end - start + 1)
            # Reset NaN interval indicators
            region[0] = region[1] = -1

    return filled


def filter_by_time(df: pd.DataFrame, time_key
                   , from_time='0000-00-00 00:00:00', to_time='9999-01-01 00:00:00'):
    filter_index = index_of_time(df, time_key, from_time, to_time)
    return df.loc[filter_index, :].reset_index(drop=True)


def index_of_time(df: pd.DataFrame, time_key
                  , from_time='0000-00-00 00:00:00', to_time='9999-01-01 00:00:00'):
    return (df[time_key] >= from_time) & (df[time_key] <= to_time)


def SMAPE(forecast: pd.Series, actual: pd.Series):
    """
    SMAPE error for predicted array compared to array of real values
    :param forecast:
    :param actual:
    :return:
    """
    if forecast.size != actual.size:
        raise ValueError("length forecast {%s} <> {%s} actual" % (forecast.size, actual.size))
    diff = np.abs(np.subtract(forecast, actual))
    avg = (np.abs(actual) + np.abs(forecast)) / 2
    return (100 / forecast.size) * np.sum(diff / avg)


def drop_columns(df: pd.DataFrame, end_with):
    """
        Drop all columns that their name ends with end_with
    :param df:
    :param end_with:
    :return:
    """
    df = df[df.columns.drop(list(df.filter(regex=end_with)))]
    return df


def fillna(df: pd.DataFrame, target, source):
    """
        Fill some columns with another columns in a dataframe
    :param df:
    :param target: array of column names
    :param source: array of column names
    :return: pd.DataFrame
    """
    for index, target in enumerate(target):
        df[target].fillna(df[source[index]], inplace=True)
    return df


def write(df: pd.DataFrame, address):
    """
        Write CSV data efficiently
    :param df:
    :param address:
    :return:
    """
    df.to_csv(address, sep=';', index=False, float_format='%.1f'
              , date_format='%Y-%m-%d %H:%M:%S', chunksize=400000)
