import unittest
import pandas as pd
from src import util


class UtilTest(unittest.TestCase):

    # test float pretty on scalar and array
    def test_pretty(self):
        self.assertEqual(util.pretty(3.1111, 2), '3.11')
        self.assertEqual(util.pretty([1.233, -3.9999], 2), ['1.23', '-4.00'])

    # test pandas data-frame normalization
    def test_normalize(self):
        normalized = util.normalize(pd.DataFrame(data={'values': [2, 3, 6]}), 2)
        # expectation: x -> 2 * (x - 2) / (6 - 2)
        self.assertEqual(normalized['values'].tolist(), [0, 0.5, 2])

