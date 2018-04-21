import const
import settings
from src.preprocess.preprocess_bj import PreProcessBJ
from src.preprocess.preprocess_ld import PreProcessLD

config = settings.config[const.DEFAULT]

config_bj = {
    const.AQ: config[const.BJ_AQ],
    const.AQ_REST: config[const.BJ_AQ_REST],
    const.AQ_LIVE: config[const.BJ_AQ_LIVE],
    const.AQ_STATIONS: config[const.BJ_AQ_STATIONS],
    const.MEO: config[const.BJ_MEO],
    const.MEO_LIVE: config[const.BJ_MEO_LIVE],
    const.OBSERVED: config[const.BJ_OBSERVED],
    const.OBSERVED_MISSING: config[const.BJ_OBSERVED_MISS],
    const.STATIONS: config[const.BJ_STATIONS],
    const.GRID_DATA: config[const.BJ_GRID_DATA],
    const.GRID_LIVE: config[const.BJ_GRID_LIVE],
    const.GRIDS: config[const.BJ_GRIDS]
}

config_ld = {
    const.AQ: config[const.LD_AQ],
    const.AQ_REST: config[const.LD_AQ_REST],
    const.AQ_LIVE: config[const.LD_AQ_LIVE],
    const.AQ_STATIONS: config[const.LD_AQ_STATIONS],
    const.OBSERVED: config[const.LD_OBSERVED],
    const.OBSERVED_MISSING: config[const.LD_OBSERVED_MISS],
    const.STATIONS: config[const.LD_STATIONS],
    const.GRID_DATA: config[const.LD_GRID_DATA],
    const.GRID_LIVE: config[const.LD_GRID_LIVE],
    const.GRIDS: config[const.LD_GRIDS]
}

append = False
pre_process_bj = PreProcessBJ(config_bj).process().append_grid(include_history=~append)\
    .fill(max_interval=3)
print('No. observed rows:', len(pre_process_bj.obs))
print('No. stations:', len(pre_process_bj.stations),
      ', for prediction:', (pre_process_bj.stations['predict'] == 1).sum())
pre_process_bj.save_features(append=append)
del pre_process_bj  # to free memory

pre_process_ld = PreProcessLD(config_ld).process().append_grid(include_history=~append)\
    .fill(max_interval=3)
print('No. observed rows:', len(pre_process_ld.obs))
print('No. stations:', len(pre_process_ld.stations),
      ', for prediction:', (pre_process_ld.stations['predict'] == 1).sum())
pre_process_ld.save_features(append=append)


