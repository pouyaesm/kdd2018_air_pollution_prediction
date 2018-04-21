import const
import settings
from src.preprocess.preprocess_bj import PreProcessBJ
from src.preprocess.preprocess_ld import PreProcessLD

# Fetch and save_features live grid data of both cities

config = settings.config[const.DEFAULT]

# -------- Configurations ------#
config_bj = {
    const.AQ_LIVE: config[const.BJ_AQ_LIVE],
    const.MEO_LIVE: config[const.BJ_MEO_LIVE],
    const.GRID_URL: config[const.BJ_GRID_URL],
    const.GRID_LIVE: config[const.BJ_GRID_LIVE]
}
config_ld = {
    const.AQ_LIVE: config[const.LD_AQ_LIVE],
    const.GRID_URL: config[const.LD_GRID_URL],
    const.GRID_LIVE: config[const.LD_GRID_LIVE]
}

pre_process_bj = PreProcessBJ(config_bj)
pre_process_ld = PreProcessLD(config_ld)

# ------- Observed Air Quality Data ---------- #
print('Fetch observed data for Beijing...')
pre_process_bj.fetch_save_live()
print('Fetch observed data for London...')
pre_process_ld.fetch_save_live()

# ------- Grid Meteorology Data ---------- #
print('Fetch grid data for Beijing...')
pre_process_bj.fetch_save_live_grid()
print('Fetch grid data for London...')
pre_process_ld.fetch_save_live_grid()

print("Done!")
