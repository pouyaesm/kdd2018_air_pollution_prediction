import const
import settings
from src.preprocess.preprocess_bj import PreProcessBJ
from src.preprocess.preprocess_ld import PreProcessLD

config = settings.config[const.DEFAULT]


# Fetch and save_features live grid data of both cities
def fetch():
    # -------- Configurations ------#
    config_bj = {
        const.AQ_LIVE: config[const.BJ_AQ_LIVE],
        const.MEO_LIVE: config[const.BJ_MEO_LIVE],
        const.GRID_URL: config[const.BJ_GRID_URL],
        const.GRID_LIVE: config[const.BJ_GRID_LIVE],
        const.GRID_FORECAST: config[const.BJ_GRID_FORECAST],
        const.BJ_AQ_URL: config[const.BJ_AQ_URL],
        const.BJ_MEO_URL: config[const.BJ_MEO_URL],
    }
    config_ld = {
        const.AQ_LIVE: config[const.LD_AQ_LIVE],
        const.GRID_URL: config[const.LD_GRID_URL],
        const.GRID_LIVE: config[const.LD_GRID_LIVE],
        const.GRID_FORECAST: config[const.LD_GRID_FORECAST],
        const.LD_AQ_URL: config[const.LD_AQ_URL],
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
    pre_process_bj.fetch_save_live_grid(city_code='bj')
    print('Fetch grid data for London...')
    pre_process_ld.fetch_save_live_grid(city_code='ld')

    print("Done!")


if __name__ == "__main__":
    fetch()
