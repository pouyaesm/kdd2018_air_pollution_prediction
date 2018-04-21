# general utilities used throughout the project
import numpy as np
import pandas as pd
import const


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


def fill(series: pd.Series, max_interval=0, inplace=False):
    """
        Replace NaN values with average of nearest non NaN neighbors
    :param max_interval: id number of consecutive NaN values > max_interval, they are not filled
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
            # do not fill NaN intervals wider than max_interval
            if max_interval <= 0 or region[1] - region[0] + 1 <= max_interval:
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
    return (1 / forecast.size) * np.sum(diff / avg)


def drop_columns(df: pd.DataFrame, end_with):
    """
        Drop all columns that their name ends with end_with
    :param df:
    :param end_with:
    :return:
    """
    df = df[df.columns.drop(list(df.filter(regex=end_with)))]
    return df


def merge_columns(df: pd.DataFrame, main: str, auxiliary: str):
    """
        Merge two columns with same prefix into one column without suffix
        For example: merge name_x and name_y into name
    :param df:
    :param main: suffix of main columns to be kept
    :param auxiliary: suffix of auxiliary columns to fill na values of corresponding main columns
    :return:
    """
    mains = set([name.split(main)[0] for name in list(df.filter(regex=main))])
    auxiliaries = set([name.split(auxiliary)[0] for name in list(df.filter(regex=auxiliary))])
    shared = list(mains.intersection(auxiliaries))  # columns shared in main and auxiliary
    only_aux = list(auxiliaries.difference(mains))

    # Fill nan values of main columns with auxiliary columns
    main_columns = [name + main for name in shared]
    aux_columns = [name + auxiliary for name in shared]
    df = fillna(df, target=main_columns, source=aux_columns)

    # Re-suffix auxiliary columns having no duplicate in main columns
    # to keep exclusively auxiliary ones in final results
    df = df.rename(columns={name + auxiliary: name + main for name in only_aux})

    # Drop auxiliary columns
    df = drop_columns(df=df, end_with=auxiliary)

    # Remove suffix from main columns
    df = df.rename(columns={col: col.split(main)[0] for col in df.columns})

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
              , date_format=const.T_FORMAT, chunksize=100000)


def shift(l: list):
    l.append(l.pop(0))


def nan_gap(values: list):
    """
        Average index interval created by sequential nan values
    :param values:
    :return:
    """
    last_index = len(values) - 1
    first_nan = -1
    last_nan = -1
    gap_sum = 0
    gap_count = 0
    for index, value in enumerate(values):
        if pd.isnull(value):
            if first_nan == -1:
                first_nan = index
            last_nan = index
        if first_nan != -1 and (not pd.isnull(value) or index == last_index):
            gap_sum += last_nan - first_nan + 1
            gap_count += 1
            first_nan = -1
    return gap_count, gap_sum, gap_sum / gap_count if gap_count > 0 else 0


def row_to_matrix(matrix: np.ndarray, row_split=1):
    return np.reshape(matrix, (matrix.shape[0], row_split, matrix.shape[1] // row_split))


def add_columns(df: pd.DataFrame, columns: np.ndarray, name_prefix='c'):
    for c in range(0, columns.shape[1]):
        df[name_prefix + str(c)] = pd.Series(data=columns[:, c], index=df.index)
    return df
