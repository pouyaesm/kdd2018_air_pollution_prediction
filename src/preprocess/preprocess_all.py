import const
import settings
from src.preprocess.preprocess_bj import PreProcessBJ
from src.preprocess.preprocess_ld import PreProcessLD

pre_process_bj = PreProcessBJ(settings.config[const.DEFAULT]).process().fill()
print('No. observed rows:', len(pre_process_bj.obs))
print('No. stations:', len(pre_process_bj.stations),
      ', for prediction:', (pre_process_bj.stations['predict'] == 1).sum())
pre_process_bj.save()

pre_process_ld = PreProcessLD(settings.config[const.DEFAULT]).process().fill()
print('No. observed rows:', len(pre_process_ld.obs))
print('No. stations:', len(pre_process_ld.stations),
      ', for prediction:', (pre_process_ld.stations['predict'] == 1).sum())
pre_process_ld.save()

