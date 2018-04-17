import pandas as pd
import numpy as np
from src import util
import const
import settings

config = settings.config[const.DEFAULT]

# observed = pd.read_csv(config[const.BJ_OBSERVED], sep=';', low_memory=True)
# stations = pd.read_csv(config[const.BJ_STATIONS], sep=';', low_memory=True)
observed = pd.read_csv(config[const.LD_OBSERVED], sep=';', low_memory=True)
stations = pd.read_csv(config[const.LD_STATIONS], sep=';', low_memory=True)

observed[const.TIME] = pd.to_datetime(observed[const.TIME], format=const.T_FORMAT, utc=True)
observed['hour_delta'] = observed.groupby(const.ID)[const.TIME].diff().fillna(0).astype('timedelta64[h]')
has_hour_gap = observed['hour_delta'] > 1
observed['has_hour_gap'] = has_hour_gap.astype(np.int32)
hour_gaps_per_station = observed.groupby(const.ID, as_index=False).agg({'has_hour_gap': 'sum'})
gaps = observed.loc[has_hour_gap, :]
print('Number of rows with hour gaps:', observed['has_hour_gap'].sum())

# Drop columns that are not important for missing values
observed.drop(columns=[const.ID, const.TIME, 'hour_delta', 'has_hour_gap'], inplace=True)

print("Missing value gaps for:")
for column in observed.columns:
    gap_count, _, gap_avg = util.nan_gap(observed[column].tolist())
    print(" {column}: count: {count}, E[length]: {avg}"
          .format(column=column, count=gap_count, avg=gap_avg))




