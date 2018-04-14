import const
import settings
from src.preprocess.preprocess_grid import PreProcessGrid
from src.preprocess.preprocess_bj import PreProcessBJ
from src.preprocess.preprocess_ld import PreProcessLD

# Fetch and save live grid data of both cities

config = settings.config[const.DEFAULT]

# ------- Observed Air Quality Data ---------- #
print('Fetch observed data for Beijing...')
pre_process_bj = PreProcessBJ(config).fetch_save_all_live()
print('Fetch observed data for London...')
pre_process_ld = PreProcessLD(config).fetch_save_all_live()
# ------- Grid Meteorology Data ---------- #
print('Fetch grid data for Beijing...')
pre_process_grid_bj = PreProcessGrid({
    const.GRID_URL: config[const.BJ_GRID_URL],
    const.GRID_LIVE: config[const.BJ_GRID_LIVE]
}).fetch_save_all_live()
print('Fetch grid data for London...')
pre_process_grid_ld = PreProcessGrid({
    const.GRID_URL: config[const.LD_GRID_URL],
    const.GRID_LIVE: config[const.LD_GRID_LIVE]
}).fetch_save_all_live()

print("Done!")
