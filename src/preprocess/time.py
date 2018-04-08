import const
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class Time:
    """
        Groups data into specific time intervals, then returns aggregated values of
        those with f(time) = value; both intervals and matching rows depend on the given mode
        modes:
            h: hour of day [0, 23]
            dw: day of week [0, 6], where 0: sunday, 6: saturday
            w: week of month [0, 3]
            m: month of year [1, 12]
            s: season of year [1, 3]
        dt.<fields>: https://pandas.pydata.org/pandas-docs/stable/api.html#datetimelike-properties
        Prepare time series data for learning
    """
    @staticmethod
    def time(ts: pd.DataFrame, mode, value=None, time_key='time', agg_op=None):
        agg_op = {'value': 'mean'} if agg_op is None else agg_op
        sample = ts.copy()
        time = pd.to_datetime(ts[time_key])
        if mode == 'h':
            sample[time_key] = time.dt.strftime("%Y-%m-%d %H:00:00")  # coarse-grain the time to hour of day
            matches_value = time.dt.hour == value
        elif mode == '3h':
            time = time.apply(lambda dt: Time.round_hour(dt, 3))
            sample[time_key] = time.dt.strftime("%Y-%m-%d %H:00:00")
            matches_value = time.dt.hour == value
        elif mode == '6h':
            time = time.apply(lambda dt: Time.round_hour(dt, 6))
            sample[time_key] = time.dt.strftime("%Y-%m-%d %H:00:00")
            matches_value = time.dt.hour == value
        elif mode == 'dw':
            sample[time_key] = time.dt.strftime("%Y-%W-%w")  # coarse-grain the time to day of week
            # in pandas dayofweek is 0: monday, 6: sunday,
            # unlike strftime where 0: sunday, 6:saturday
            matches_value = time.dt.dayofweek == ((value + 6) % 7)
        elif mode == 'm':
            sample[time_key] = time.dt.strftime("%Y-%m")  # coarse-grain the time to month
            matches_value = time.dt.month == value
        elif mode == 's':
            sample[time_key] = np.mod(time.dt.strftime('%Y-%m').astype(int), 4)

        # sample those with matching time values
        sample = sample.loc[matches_value, :] if value is not None else sample
        # aggregate values based on mode
        sample = sample.groupby([time_key], as_index=False).agg(agg_op)
        sample.reset_index(drop=True, inplace=True)
        return sample

    @staticmethod
    def round_hour(time: datetime, hours):
        return time.replace(hour=time.hour - time.hour % hours, minute=0, second=0)
