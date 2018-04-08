import unittest
import numpy as np
import pandas as pd
import numpy.testing as np_test
from src import reform


class UtilTest(unittest.TestCase):

    # test splitting a time-series into (x, y) blocks for prediction
    def test_window(self):
        # test simple window
        windowed = reform.window(values=pd.Series([1, 2, 3, 4, 5, 6]), window_size=3, step=2)
        expected = np.array([[1, 2, 3], [3, 4, 5]])
        np_test.assert_array_equal(windowed, expected)

        # test windowing into (x, y) for prediction
        x, y = reform.window_for_predict(values=pd.Series([1, 2, 3, 4])
                                           , x_size=2, y_size=1, step=1)
        expected = {'x': np.array([[1, 2], [2, 3]]), 'y': np.array([[3], [4]])}
        np_test.assert_array_equal(x, expected['x'])
        np_test.assert_array_equal(y, expected['y'])
