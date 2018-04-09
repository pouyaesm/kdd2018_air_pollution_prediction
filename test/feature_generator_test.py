import unittest
import numpy as np
import pandas as pd
import const
import numpy.testing as np_test
import pandas.util.testing as pd_test
import pandas.util.testing as pd_test
from src.preprocess import reform
from src.preprocess.feature_generator import FeatureGenerator


class FeatureGeneratorTest(unittest.TestCase):

    def test_basic_features(self):
        base_dir = "E:\\Projects\\KDD2018\\KDD2018Predict\\test\\data\\"
        hours = 10  # number of values for input/output
        fg = FeatureGenerator({
            const.OBSERVED: base_dir + "beijing_observed_sample.csv",
            const.STATIONS: base_dir + "beijing_stations_sample.csv"
        }, hour_x=hours, hour_y=hours)
        fg.load().basic().sample(20)
        self.assertEqual(20, fg.features.shape[0])  # number of data points
        self.assertEqual(3 + hours * 2, fg.features.shape[1])  # number of columns

