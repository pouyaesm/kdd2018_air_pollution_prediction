import settings
import const
from src.methods.hybrid import Hybrid


def generate():
    config = settings.config[const.DEFAULT]
    cases = {
        'BJ':[
             'PM2.5',
             # 'PM10',
             # 'O3'
        ],
        'LD': [
            # 'PM2.5',
            # 'PM10'
        ]
    }

    for city in cases:
        for pollutant in cases[city]:
            # For low values of pollutants MAE works better than SMAPE!
            # So for all pollutants of london and O3 of beijing we use MAE
            cfg = {
                const.CITY: city,
                const.MODEL_DIR: config[const.MODEL_DIR],
                const.FEATURE_DIR: config[const.FEATURE_DIR],
                const.POLLUTANT: pollutant,
                const.FEATURE: getattr(const, city + '_' + pollutant.replace('.', '') + '_'),
                const.STATIONS: config[getattr(const, city + '_STATIONS')],
                const.LOSS_FUNCTION: const.MEAN_PERCENT,
                const.TEST_FROM: '18-04-01 23',
                const.TEST_TO: '18-04-26 23',
                const.CHUNK_COUNT: 8,
                const.TIME_STEPS: 12
            }
            # LSTM(cfg, time_steps=48).load_model().test()
            Hybrid(cfg).load_model(mode='best').test()
            print(city, pollutant, 'done!')


if __name__ == "__main__":
    generate()
