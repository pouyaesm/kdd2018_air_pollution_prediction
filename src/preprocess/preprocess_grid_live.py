import const
import settings
from src.preprocess.preprocess_grid import PreProcessGrid

# Fetch and save live grid data of both cities

config = settings.config[const.DEFAULT]
print('Fetch grid data for Beijing...')
pre_process_bj = PreProcessGrid({
    const.GRID_URL: config[const.BJ_GRID_URL],
    const.GRID_LIVE: config[const.BJ_GRID_LIVE]
}).fetch_save_all_live()
print('Fetch grid data for London...')
pre_process_ld = PreProcessGrid({
    const.GRID_URL: config[const.LD_GRID_URL],
    const.GRID_LIVE: config[const.LD_GRID_LIVE]
}).fetch_save_all_live()

print("Done!")
