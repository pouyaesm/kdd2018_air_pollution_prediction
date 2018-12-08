import settings
import const
from src.methods.hybrid import Hybrid
from src.feature_generators.hybrid_fg import HybridFG


def generate():
    config = settings.config[const.DEFAULT]
    sub_folder = ""
    cases = {
        'BJ': [
             # 'PM2.5',
             # 'PM10',
             # 'O3'
        ],
        'LD': [
            'PM2.5',
            # 'PM10'
        ]
    }

    for city in cases:
        for pollutant in cases[city]:
            # For low values of pollutants MAE works better than SMAPE!
            # So for all pollutants of london and O3 of beijing we use MAE
            cfg = {
                const.CITY: city,
                const.MODEL_DIR: config[const.MODEL_DIR] + sub_folder,
                const.FEATURE_DIR: config[const.FEATURE_DIR],
                const.POLLUTANT: pollutant,
                const.STATIONS: config[getattr(const, city + '_STATIONS')],
                const.LOSS_FUNCTION: const.MEAN_PERCENT,
                const.TEST_FROM: '18-04-01 23',
                const.TEST_TO: '18-05-01 00',
            }
            cfg.update(HybridFG.get_size_config(city=city, key='05-02'))
            Hybrid(cfg).load_model(mode='best').test()
            print(city, pollutant, 'done!')


if __name__ == "__main__":
    generate()
