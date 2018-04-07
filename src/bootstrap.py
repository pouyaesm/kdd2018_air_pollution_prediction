import settings
import const
from src.preprocess.preprocess_bj import PreProcessBJ
from src.preprocess.preprocess_ld import PreProcessLD

if __name__ == "__main__":
    pre_process_bj = PreProcessBJ(settings.config[const.DEFAULT]).process().save()
    pre_process_ld = PreProcessLD(settings.config[const.DEFAULT]).process().save()
