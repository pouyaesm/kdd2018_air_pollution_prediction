import unittest
import const
from src.preprocess.preprocess_bj import PreProcessBJ


class PreProcessTest(unittest.TestCase):

    # test load_model and pre-processing data
    def test_pre_process(self):
        base_dir = "E:\\Projects\\KDD2018\\KDD2018Predict\\test\\data\\"
        config = {
            const.BJ_AQ: base_dir + "beijing_17_18_aq_sample.csv",
            const.BJ_AQ_REST: base_dir + "beijing_201802_201803_aq_sample.csv",
            const.BJ_AQ_STATIONS: base_dir + "beijing_AirQuality_Stations_sample.csv",
            const.BJ_MEO: base_dir + "beijing_17_18_meo_sample.csv",
            const.BJ_READ_LIVE: 0,
        }
        pre_process = PreProcessBJ(config).process().fill()
        self.assertEqual(9, len(pre_process.obs))
        # Expecting 9 NaN values after fill(), 6 for position, 3 for weather
        self.assertEqual(9, len(pre_process.obs.isnull().values))
