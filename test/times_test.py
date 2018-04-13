import unittest
import pandas as pd
import numpy.testing as np_test
import pandas.util.testing as pd_test
from src.preprocess import times


class UtilTest(unittest.TestCase):

    # round hours to a given interval
    def test_round_hour(self):
        dt = pd.to_datetime('2017-01-01 23:30:00', utc=True)
        # round to 3 hours
        rounded = times.round_hour(dt, 3)
        self.assertEqual('21:00:00', rounded.strftime('%H:%M:%S'))
        # round to 6 hours
        rounded = times.round_hour(dt, 6)
        self.assertEqual('18:00:00', rounded.strftime('%H:%M:%S'))
        # round to 12 hours
        rounded = times.round_hour(dt, 12)
        self.assertEqual('12:00:00', rounded.strftime('%H:%M:%S'))
        # round to 24 hours
        rounded = times.round_hour(dt, 24)
        self.assertEqual('00:00:00', rounded.strftime('%H:%M:%S'))

    # test time sampling for different time modes
    @staticmethod
    def test_aggregate_time():
        # sample all hours
        df = pd.DataFrame(data={
            'time': ['2017-01-01 18:20:00', '2017-01-01 18:40:00', '2017-02-07 23:00:01'],
            'value': [1, 2, 3]
        })
        sample = times.group(ts=df, mode='h')
        expected = pd.DataFrame(data={'time': ['2017-01-01 18:00:00', '2017-02-07 23:00:00'],
                                      'value': [1.5, 3]})
        pd_test.assert_frame_equal(sample, expected)
        # sample all 3 hours
        # 23:00 is expected to become 21:00
        sample = times.group(ts=df, mode='3h')
        expected = pd.DataFrame(data={'time': ['2017-01-01 18:00:00', '2017-02-07 21:00:00'],
                                      'value': [1.5, 3]})
        pd_test.assert_frame_equal(sample, expected)
        # sample a specific month = 2
        df = pd.DataFrame(data={
            'time': ['2016-02-01', '2017-01-01', '2017-02-07', '2017-02-07'],
            'value': [2, 1, 2, 3]})
        sample = times.group(ts=df, mode='m', value=2)
        expected = pd.DataFrame(data={'time': ['2016-02', '2017-02'], 'value': [2, 2.5]})
        pd_test.assert_frame_equal(sample, expected)
        # sample a specific day of week, 2017-01-01 is sunday (0)
        df = pd.DataFrame(data={
            'time': ['2017-01-01 12:00', '2017-01-01 14:00', '2017-01-06 10:00'], 'value': [1, 2, 3]})
        sample = times.group(ts=df, mode='dw', value=0)
        expected = pd.DataFrame(data={'time': ['2017-00-0'], 'value': [1.5]})
        pd_test.assert_frame_equal(sample, expected)

    @staticmethod
    def test_filter_by_time():
        df = pd.DataFrame(data={'time': ['2019-01-01', '2020-01-01']})
        df['time'] = pd.to_datetime(df['time'])
        filtered = times.select(df, 'time', from_time='2019-01-01', to_time='2019-01-30')
        expected = pd.DataFrame(data={'time': [pd.datetime(2019, 1, 1)]})
        pd_test.assert_frame_equal(filtered, expected)

    @staticmethod
    def test_group_from():
        # test grouping values backward or forward from a given value and given hours unit
        yr = '2018-01-01 '
        time = pd.to_datetime([yr + ' 12', yr + ' 15', yr + '16', yr + '17'], utc=True).tolist()
        value = [2, 2, 3, 4]
        # expected to group values into 9:00, 12:00 and 15:00 (3 steps of hour = 3)
        # where 9:00 is expected to be the repetition of 12:00
        grouped_back = times.group_from(time, value, index=2, step=-3, hours=3)
        np_test.assert_array_equal(grouped_back, [2, 2, 2.5])
        # expected to group forward into 150:00
        grouped_back = times.group_from(time, value, index=2, step=3, hours=3)
        np_test.assert_array_equal(grouped_back, [3.5, 3.5, 3.5])

    @staticmethod
    def test_group_at():
        # test grouping values backward or forward from a given value and given hours unit
        yr = '2018-01-01 '
        time = pd.to_datetime([yr + ' 12', yr + ' 15', yr + '16', yr + '17'], utc=True).tolist()
        value = [2, 2, 3, 4]
        # expected to group values into 12:00 and 15:00 (3 steps of hour = 3)
        grouped_forward = times.group_at(time, value, hours=3)
        np_test.assert_array_equal(x=[2, 2, 2.5, 3], y=grouped_forward)
