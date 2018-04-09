import unittest
import pandas as pd
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
