import settings
import const
from src.methods.lstm import LSTM

config = settings.config[const.DEFAULT]
pollutant = 'PM2.5'
features_bj = config[getattr(const, 'BJ_' + pollutant.replace('.', '') + '_')]
features_ld = config[getattr(const, 'LD_' + pollutant.replace('.', '') + '_')]
# For low values of pollutants MAE works better than SMAPE!
# So for all pollutants of london and O3 of beijing we use MAE
config_bj = {
    const.FEATURES: features_bj,
    const.LOSS_FUNCTION: const.MEAN_ABSOLUTE
}
config_ld = {
    const.FEATURES: features_ld,
    const.LOSS_FUNCTION: const.MEAN_ABSOLUTE
}
lstm = LSTM(config_bj, time_steps=48).load_model().test()
