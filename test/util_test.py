import unittest
import numpy as np
import pandas as pd
import pandas.util.testing as pd_test
import numpy.testing as np_test
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

    # test dropping columns ending with a specific string
    def test_drop_columns(self):
        df = pd.DataFrame(data={'col_1': [1, 2], 'col_2': [2, 3]})
        expected = pd.DataFrame(data={'col_1': [1, 2]})
        pd_test.assert_frame_equal(util.drop_columns(df, end_with='_2'), expected)

    # test filling nan columns with another columns
    def test_fillna(self):
        df = pd.DataFrame(data={'col_1': [np.nan], 'col_2': [10]}, dtype=np.int64)
        filled = util.fillna(df, target=['col_1'], source=['col_2'])
        expected = pd.DataFrame(data={'col_1': [10], 'col_2': [10]})
        pd_test.assert_frame_equal(filled, expected)

    # test filling of missing values (NaNs) by nearest non NaN neighbors
    def test_fill(self):
        series = pd.Series([np.nan, 1, np.nan, np.nan, 3, np.nan], index=[10, 11, 12, 13, 14, 15])
        util.fill(series, inplace=True)
        expected = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], index=[10, 11, 12, 13, 14, 15])
        pd_test.assert_series_equal(series, expected)

    # test SMAPE evaluation criterion
    def test_SMAPE_error(self):
        forecast = pd.Series([1, 2, 3, 4])
        actual = pd.Series([0, 2, 3, 6])
        # 100 / 4 * (1 / 0.5 + 2 / 5)
        error = util.SMAPE(forecast, actual)
        self.assertEqual(error, 60)

