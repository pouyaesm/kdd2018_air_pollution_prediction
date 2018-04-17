# Plots time series of pollutants (PM2.5, PM10, O3) per station

import settings
import const
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Access default configurations
config = settings.config[const.DEFAULT]


# Turn off plot interactive mode to let IDE display the plots
plt.interactive(False)

# Read cleaned data
df = pd.read_csv(config[const.BJ_OBSERVED], delimiter=';', low_memory=False).sample(frac=0.05)

# ---------------------------------
# Pollutants' time-series
# ---------------------------------
pollutants = ['PM2.5', 'PM10', 'O3']

fig_pollutants, axes_pollutants = plt.subplots(nrows=len(pollutants), ncols=1)
# dedicate 20% of right side of plot to legend
fig_pollutants.tight_layout()
fig_pollutants.subplots_adjust(right=0.75)

for index, pollutant in enumerate(pollutants):
    for station_id, group in df.groupby(['station_id']):
        axes_pollutants[index].plot(pd.to_datetime(group['utc_time'], format=const.T_FORMAT),
                                    group[pollutant], 'o', label=station_id, alpha=0.5)
        axes_pollutants[index].set_title(pollutant)
        axes_pollutants[index].set_yscale('log')  # plot values in log for visibility of smaller values

# place the stations legend to the right of plots in two columns
axes_pollutants[1].legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=2)

# ---------------------------------
# Weather measurements' time-series
# ---------------------------------
measurements = ['temperature', 'humidity', 'pressure', 'wind_speed']

fig_weather, axes_weather = plt.subplots(nrows=len(measurements), ncols=1)
fig_weather.tight_layout()
fig_weather.subplots_adjust(right=0.75)

for index, measurement in enumerate(measurements):
    for station_id, group in df.groupby(['station_id']):
        axes_weather[index].plot(pd.to_datetime(group['utc_time'], format=const.T_FORMAT),
                                 group[measurement], 'o', label=station_id, alpha=0.5)
        axes_weather[index].set_title(measurement)

# place the stations legend to the right of plots in two columns
axes_weather[1].legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=2)
plt.show()
