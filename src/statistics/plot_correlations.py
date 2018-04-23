import settings
import const
import pandas as pd
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})  # to prevent labels going out of plot!
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Access default configurations
config = settings.config[const.DEFAULT]

# Read cleaned data
df = pd.read_csv(config[const.BJ_OBSERVED], delimiter=';', low_memory=False)

corr = df.corr()
fig, ax = plt.subplots()
cax = ax.matshow(corr, vmin=-1, vmax=1, cmap='hot')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
fig.colorbar(cax)  # correspondence of colors to values

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# cmap = cm.get_cmap('jet', 30)
# cax = ax1.imshow(corr, interpolation="nearest", cmap=cmap)
# ax1.grid(True)
# plt.title('Feature Correlation')
# ax1.set_xticklabels(corr.columns,fontsize=10)
# ax1.set_yticklabels(corr.columns,fontsize=10)
# # Add colorbar, make sure to specify tick locations to match desired ticklabels
# fig.colorbar(cax, ticks=[-1, -.75, -.5, -.25, 0, .25, 0.5, 0.75, 1])
plt.show()

