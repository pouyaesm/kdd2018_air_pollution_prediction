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
