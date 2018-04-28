import settings
import const
import pandas as pd
from IPython.display import display
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})  # to prevent labels going out of plot!
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Access default configurations
config = settings.config[const.DEFAULT]

# Turn off plot interactive mode to let IDE display the plots
plt.interactive(False)

# df = pd.read_csv(config[const.BJ_OBSERVED], delimiter=';', low_memory=False)
df = pd.read_csv(config[const.LD_OBSERVED], delimiter=';', low_memory=False)
print('No. data rows:', len(df.index))

# Basic statistics of air quality
display(df[const.PM25].describe())
print('_________________')
display(df[const.PM10].describe())
print('_________________')
if const.O3 in df.columns:
    display(df[const.O3].describe())
    print('_________________')

# Distribution of three pollutants of interest
fig, axes = plt.subplots(nrows=1, ncols=3)
fig.tight_layout()

binned_PM25 = pd.cut(df[const.PM25], bins=np.linspace(0, 200, 10), include_lowest=True)
binned_PM25.value_counts(sort=False).plot(ax=axes[0], title=const.PM25, fontsize=13, kind='bar', color='maroon')

binned_PM10 = pd.cut(df[const.PM10], bins=np.linspace(0, 250, 10), include_lowest=True)
binned_PM10.value_counts(sort=False).plot(ax=axes[1], title=const.PM10, fontsize=13, kind='bar', color='firebrick')

if const.O3 in df.columns:
    binned_O3 = pd.cut(df[const.O3], bins=np.linspace(0, 150, 10), include_lowest=True)
    binned_O3.value_counts(sort=False).plot(ax=axes[2], title=const.O3, fontsize=13, kind='bar', color='tomato')

# Basic statistics of weather
display(df['temperature'].describe())
print('_________________')
display(df['pressure'].describe())
print('_________________')
display(df['humidity'].describe())
print('_________________')
display(df['wind_speed'].describe())
print('_________________')
display(df['wind_direction'].describe())
print('_________________')

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
