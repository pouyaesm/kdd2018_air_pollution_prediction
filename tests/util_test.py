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

    # test splitting a time-series into (x, y) blocks for prediction
    def test_window(self):
        # test simple window
        windowed = util.window(values=pd.Series([1, 2, 3, 4, 5, 6]), window_size=3, step=2)
        expected = np.array([[1, 2, 3], [3, 4, 5]])
        np_test.assert_array_equal(windowed, expected)

        # test windowing into (x, y) for prediction
        windowed = util.window_for_predict(values=pd.Series([1, 2, 3, 4])
                                           , x_size=2, y_size=1, step=1)
        expected = {'x': np.array([[1, 2], [2, 3]]), 'y': np.array([[3], [4]])}
        np_test.assert_array_equal(windowed['x'], expected['x'])
        np_test.assert_array_equal(windowed['y'], expected['y'])

    # test time sampling for different time modes
    def test_sample_time(self):
        # sample all hours
        df = pd.DataFrame(data={
            'time': ['2017-01-01 18:20:00', '2017-01-01 18:40:00', '2017-02-07 23:00:01'],
            'value': [1, 2, 3]
        })
        sample = util.sample_time(time_series=df, mode='h')
        expected = pd.DataFrame(data={'time': ['2017-01-01 18:00:00', '2017-02-07 23:00:00'],
                                      'value': [1.5, 3]})
        pd_test.assert_frame_equal(sample, expected)
        # sample a specific month = 2
        df = pd.DataFrame(data={
            'time': ['2016-02-01', '2017-01-01', '2017-02-07', '2017-02-07'],
            'value': [2, 1, 2, 3]})
        sample = util.sample_time(time_series=df, mode='m', value=2)
        expected = pd.DataFrame(data={'time': ['2016-02', '2017-02'], 'value': [2, 2.5]})
        pd_test.assert_frame_equal(sample, expected)
        # sample a specific day of week, 2017-01-01 is sunday (0)
        df = pd.DataFrame(data={
            'time': ['2017-01-01 12:00', '2017-01-01 14:00', '2017-01-06 10:00'], 'value': [1, 2, 3]})
        sample = util.sample_time(time_series=df, mode='dw', value=0)
        expected = pd.DataFrame(data={'time': ['2017-00-0'], 'value': [1.5]})
        pd_test.assert_frame_equal(sample, expected)
