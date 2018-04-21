import settings
import const
from src.methods.lstm import LSTM

config = settings.config[const.DEFAULT]
cases = {
    'BJ':[
         'PM2.5',
         'PM10',
         'O3'
    ],
    'LD': [
        'PM2.5',
        'PM10'
    ]
}

for city in cases:
    for pollutant in cases[city]:
        # For low values of pollutants MAE works better than SMAPE!
        # So for all pollutants of london and O3 of beijing we use MAE
        cfg = {
            const.MODEL_DIR: config[const.MODEL_DIR],
            const.FEATURE_DIR: config[const.FEATURE_DIR],
            const.FEATURE: getattr(const, city + '_' + pollutant.replace('.', '') + '_'),
            const.LOSS_FUNCTION: const.MEAN_ABSOLUTE
        }
        LSTM(cfg, time_steps=48).load_model().test()
        print(city, pollutant, 'done!')
