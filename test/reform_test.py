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
        values = [1, 2, 3, 4, 5]
        yr = '2018-01-01 '
        times = pd.to_datetime([yr + '12:00:00', yr + '13:00:00', yr + '14:00:00', yr + '15:00:00'],
                               utc=True).tolist()
        t, x, y = reform.split_by_hours(times, values, hours_x=2, hours_y=2)
        # day of week (1: monday), hour, value of two hours
        expected_x = [[1, 2], [2, 3]]
        expected_y = [[3, 4], [4, 5]]
        expected_t = pd.to_datetime(pd.Series(
            [yr + '12:00:00', yr + '13:00:00']
        ), utc=True).tolist()
        np_test.assert_array_equal(expected_t, t)
        np_test.assert_array_equal(expected_x, x)
        np_test.assert_array_equal(expected_y, y)

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
