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

    # merge values of two columns with similar names to maximum one
    def test_merge_columns(self):
        df = pd.DataFrame(data={'id_1': [1, 3], 'name_1': ['a', np.nan], 'name_2': ['a', 'b'],
                                'rate_2': [3, 4]})
        expected = pd.DataFrame(data={'id': [1, 3], 'name': ['a', 'b'], 'rate': [3, 4]})
        merged = util.merge_columns(df, main='_1', auxiliary='_2')
        pd_test.assert_frame_equal(left=expected, right=merged)

    # test filling nan columns with another columns
    def test_fillna(self):
        df = pd.DataFrame(data={'col_1': [np.nan], 'col_2': [10]}, dtype=np.int64)
        filled = util.fillna(df, target=['col_1'], source=['col_2'])
        expected = pd.DataFrame(data={'col_1': [10], 'col_2': [10]})
        pd_test.assert_frame_equal(filled, expected)

    # test filling of missing values (NaNs) by nearest non NaN neighbors
    def test_fill(self):
        series = pd.Series([np.nan, 1, np.nan, np.nan, 3, np.nan, np.nan, np.nan, 5, np.nan],
                           index=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        util.fill(series, max_interval=2, inplace=True)
        expected = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0, np.nan, np.nan, np.nan, 5.0, 5.0],
                             index=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        pd_test.assert_series_equal(series, expected)

    # test SMAPE evaluation criterion
    def test_SMAPE_error(self):
        forecast = pd.Series([1, 2, 3, 4])
        actual = pd.Series([0, 2, 3, 6])
        # 1 / 4 * (1 / 0.5 + 2 / 5)
        error = util.SMAPE(forecast, actual)
        self.assertEqual(error, 0.6)

    # test the relation between dataframe and numpy 2d structures (matrices)
    def test_dataframe_numpy_convertion(self):
        df = pd.DataFrame(data={'a': [1, 2], 'b': [3, 4]})
        df_to_np = df.values.reshape(df.size)
        array = np.array([[1, 3], [2, 4]]).reshape(df.size)
        np_test.assert_array_equal(df_to_np, array)

    @staticmethod
    def test_shift():
        values = [1, 2, 3]
        util.shift(values)
        np_test.assert_array_equal(x=[2, 3, 1], y=values)

    def test_mean_nan_interval(self):
        values = [np.nan, np.nan, 1, 2, np.nan, 3, np.nan, np.nan]
        gap_count, gap_sum, gap_avg = util.nan_gap(values)
        # expected: (2 + 1 + 2) / 3
        self.assertEqual(first=5, second=gap_sum)
        self.assertEqual(first=3, second=gap_count)
        self.assertEqual(first=5/3, second=gap_avg)

        values = [np.nan, np.nan, "a", "b"]
        _, gap_sum, _ = util.nan_gap(values)
        self.assertEqual(first=2, second=gap_sum)
