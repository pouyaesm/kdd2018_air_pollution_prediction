import unittest
import numpy as np
import pandas as pd
import const
import numpy.testing as np_test
import pandas.util.testing as pd_test
from src.preprocess import reform


class UtilTest(unittest.TestCase):

    @staticmethod
    def test_window():
        """
            Test splitting a time-series into (x, y) blocks for prediction
        :return:
        """
        # test simple window
        windowed = reform.window(values=pd.Series([1, 2, 3, 4, 5, 6]), window_size=3, step=2)
        expected = np.array([[1, 2, 3], [3, 4, 5]])
        np_test.assert_array_equal(windowed, expected)

        # test windowing into (x, y) for prediction
        x, y = reform.window_for_predict(values=pd.Series([1, 2, 3, 4]),
                                         x_size=2, y_size=1, step=1)
        expected = {'x': np.array([[1, 2], [2, 3]]), 'y': np.array([[3], [4]])}
        np_test.assert_array_equal(x, expected['x'])
        np_test.assert_array_equal(y, expected['y'])

    @staticmethod
    def test_split_hours():
        """
            Test splitting a time series into hours
        :return:
        """
        values = pd.Series([1, 2, 3, 4, 5])
        y = '2018-01-01 '
        times = pd.to_datetime(pd.Series(
            [y + '12:00:00', y + '13:00:00', y + '14:00:00', y + '15:00:00']
        ), utc=True)
        split_x, split_y, time_x = reform.split_by_hours(times, values, hours_x=2, hours_y=2)
        # day of week (1: monday), hour, value of two hours
        expected_split_x = np.array([[1, 12, 1, 2], [1, 13, 2, 3]])
        expected_split_y = np.array([[3, 4], [4, 5]])
        np_test.assert_array_equal(expected_split_x, split_x)
        np_test.assert_array_equal(expected_split_y, split_y)

    @staticmethod
    def test_group_by_station():
        data = pd.DataFrame(data={const.ID: [1, 2, 3], 'value': [5, 6, 7]})
        stations = pd.DataFrame(data={const.ID: [1, 2, 3], const.PREDICT: [1, 0, 1]})
        grouped = reform.group_by_station(ts=data, stations=stations)
        expected = {
            1: pd.DataFrame(data={const.ID: [1], 'value': [5]}),
            3: pd.DataFrame(data={const.ID: [3], 'value': [7]}),
        }
        pd_test.assert_frame_equal(expected[1], grouped[1])
        pd_test.assert_frame_equal(expected[3], grouped[3])
