import settings
import const
import pandas as pd
from IPython.display import display
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Access default configurations
config = settings.config[const.DEFAULT]

# Turn off plot interactive mode to let IDE display the plots
plt.interactive(False)

df = pd.read_csv(config[const.CLEAN_DATA], delimiter=';', low_memory=False)
print('No. data rows:', len(df.index))

# Basic statistics of air quality
display(df['PM2.5'].describe())
display(df['PM10'].describe())
display(df['O3'].describe())

# Distribution of three pollutants of interest
fig, axes = plt.subplots(nrows=1, ncols=3)
fig.tight_layout()

binned_PM25 = pd.cut(df['PM2.5'], bins=np.linspace(0, 200, 10), include_lowest=True)
binned_PM25.value_counts(sort=False).plot(ax=axes[0], title='PM 2.5', fontsize=13, kind='bar', color='maroon')

binned_PM25 = pd.cut(df['PM10'], bins=np.linspace(0, 250, 10), include_lowest=True)
binned_PM25.value_counts(sort=False).plot(ax=axes[1], title='PM 10', fontsize=13, kind='bar', color='firebrick')

binned_PM25 = pd.cut(df['O3'], bins=np.linspace(0, 150, 10), include_lowest=True)
binned_PM25.value_counts(sort=False).plot(ax=axes[2], title='O3', fontsize=13, kind='bar', color='tomato')

# Basic statistics of weather
display(df['temperature'].describe())
display(df['wind_speed'].describe())
display(df['wind_direction'].describe())
display(df['pressure'].describe())
display(df['humidity'].describe())

# Distribution of major weather indicators like temperature and wind
fig1, axes1 = plt.subplots(nrows=1, ncols=3)
fig2, axes2 = plt.subplots(nrows=1, ncols=3)
fig1.tight_layout()
fig2.tight_layout()

df['weather'].value_counts(sort=False)\
    .plot(ax=axes1[0], title='weather', fontsize=13, kind='bar', color='skyblue')

binned_temperature = pd.cut(df['temperature'], bins=np.linspace(-20, 40, 13), include_lowest=True)
binned_temperature.value_counts(sort=False)\
    .plot(ax=axes1[1], title='temperature', fontsize=13, kind='bar', color='tomato')

binned_humidity = pd.cut(df['humidity'], bins=np.linspace(0, 100, 11), include_lowest=True)
binned_humidity.value_counts(sort=False)\
    .plot(ax=axes1[2], title='humidity', fontsize=13, kind='bar', color='navy')

binned_speed = pd.cut(df['wind_speed'], bins=np.linspace(0, 10, 21), include_lowest=True)
binned_speed.value_counts(sort=False)\
    .plot(ax=axes2[0], title='wind speed', fontsize=13, kind='bar', color='darkorange')

binned_direction = pd.cut(df['wind_direction'], bins=np.linspace(0, 360, 25), include_lowest=True)
binned_direction.value_counts(sort=False)\
    .plot(ax=axes2[1], title='wind direction', fontsize=13, kind='bar', color='orange')

binned_pressure = pd.cut(df['pressure'], bins=np.linspace(990, 1040, 11), include_lowest=True)
binned_pressure.value_counts(sort=False)\
    .plot(ax=axes2[2], title='pressure', fontsize=13, kind='bar', color='firebrick')

plt.show()
