import unittest
import pandas as pd
import numpy as np
from src.feature_generators.fg import FG

class UtilTest(unittest.TestCase):

    # test normalization, de-normalization
    @staticmethod
    def test_normalize():
        values = pd.Series(data=[60, 65, 70])
        normal = FG.normalize(values, city='BJ', name='PM2.5')
        expected = pd.Series(data=[-0.014925,  0.059701,  0.134328])
        np.allclose(a=expected, b=normal)
        de_normal = FG.de_normalize(values=normal, city='BJ', name='PM2.5')
        np.allclose(a=values, b=de_normal)
