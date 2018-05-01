import unittest
import pandas as pd
import numpy as np
import numpy.testing as np_test
import pandas.util.testing as pd_test
from src.preprocess import times
from datetime import datetime


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

    def test_group_at(self):
        # test grouping values backward or forward from a given value and given hours unit
        yr = '2018-01-01 '
        time = pd.to_datetime([yr + ' 12', yr + ' 15', yr + '16', yr + '17'], utc=True).tolist()
        value = [2, 2, 3, 4]
        # expected groups: 12:00, 15:00
        forward_average = times.group_at(time, value, index=2, direction=1, group_hours=3)
        self.assertEqual(first=3.5, second=forward_average)
        backward_average = times.group_at(time, value, index=2, direction=-1, group_hours=3)
        self.assertEqual(first=2.5, second=backward_average)
        single_average = times.group_at(time, value, index=0, direction=-1, group_hours=3)
        self.assertEqual(first=2, second=single_average)


    @staticmethod
    def test_group_from():
        # test grouping values backward or forward from a given value and given hours unit
        yr = '2018-01-01 '
        time = pd.to_datetime([yr + ' 12', yr + ' 15', yr + '16', yr + '17', yr + '18'], utc=True).tolist()
        value = [2, 2, 3, 4, 5]
        # expected to group (2, 2, 3) into 9:00, 12:00 and 15:00
        # where 9:00 is expected to be the repetition of 12:00
        grouped_back = times.group_from(time, value, index=2, step=-3, group_hours=3)
        np_test.assert_array_equal(grouped_back, [2, 2, 2.5])
        # expected to group (3, 4) forward into 15:00, and (5) into 18:00
        grouped_forward = times.group_from(time, value, index=2, step=2, group_hours=3)
        np_test.assert_array_equal(grouped_forward, [3.5, 5])
        # redo the test including the group boundaries
        # group boundary of 16:00 is 17:00 for backward grouping
        boundary_grouped_back = times.group_from(time, value, index=2, step=-3, group_hours=3,
                                                 whole_group=True)
        np_test.assert_array_equal(boundary_grouped_back, [2, 2, 3])
        # group boundary of 16:00 is [15:00, 17:00) for forward grouping
        boundary_grouped_back = times.group_from(time, value, index=2, step=3, group_hours=3,
                                                 whole_group=True)
        np_test.assert_array_equal(boundary_grouped_back, [3, 5, 5])


    @staticmethod
    def test_running_average():
        # test averaging values backward or forward from a given value and given hours unit
        yr = '2018-01-01 '
        time = pd.to_datetime([yr + ' 12', yr + ' 13', yr + ' 15', yr + '16', yr + '17'], utc=True).tolist()
        value = [2, np.nan, 3, 4, 5]
        # expected to forward-average values into 12:00 and 15:00 (unit: 3 hours)
        forward_averaged = times.running_average(time=time, value=value, group_hours=3)
        np_test.assert_array_equal(x=[2, 2, 3, 3.5, 4], y=forward_averaged)
        # expected to backward-average
        backward_averaged = times.running_average(time=time, value=value, group_hours=3, direction=-1)
        np_test.assert_array_equal(x=[2, 0, 4, 4.5, 5], y=backward_averaged)
        # put whole group average for each member
        whole_averaged = times.running_average(time=time, value=value, group_hours=3, whole_group=True)
        np_test.assert_array_equal(x=[2, 2, 4, 4, 4], y=whole_averaged)

    def test_group_average(self):
        yr = '2018-01-01 '
        time = pd.to_datetime([yr + ' 13', yr + '14', yr + ' 15', yr + '16', yr + '17', yr + '18'], utc=True).tolist()
        value = [2, 3, 4, np.nan, 5, 6]
        expected_group_time = pd.to_datetime([yr + ' 12', yr + '15', yr + ' 18'], utc=True).tolist()
        group_time, group_average, lookup, count = times.group_average(time=time, value=value, group_hours=3)
        # time groups: 12:00, 15:00, and 18:00
        np_test.assert_array_equal(x=expected_group_time, y=group_time)
        np_test.assert_array_equal(x=[2.5, 4.5, 6], y=group_average)
        np_test.assert_array_equal(x=[2, 2, 1], y=count)
        self.assertEqual(0, lookup[expected_group_time[0]])
        self.assertEqual(1, lookup[expected_group_time[1]])
        self.assertEqual(2, lookup[expected_group_time[2]])

    @staticmethod
    def test_split():
        yr = '2018-01-01 '
        time = pd.to_datetime([yr + ' 13', yr + '14', yr + ' 15', yr + '16', yr + '18'], utc=True).tolist()
        value = [2, 3, 4, 5, 6]
        # time groups: 12:00, 15:00, and 18:00
        # averaging from group start until given index
        split = times.split(time=time, value=value, group_hours=3, step=-3, region=(0, -1))
        expected = [[2, 2, 2], [2.5, 2.5, 2.5], [2.5, 2.5, 4], [2.5, 2.5, 4.5], [2.5, 4.5, 6]]
        np_test.assert_array_equal(x=expected, y=split)

        # time groups: 12:00, 15:00, and 18:00
        # averaging from group start until given index
        split = times.split(time=time, value=value, group_hours=3, step=3, region=(0, -1))
        expected = [[2.5, 4.5, 6], [3, 4.5, 6], [4.5, 6, 6], [5, 6, 6], [6, 6, 6]]
        np_test.assert_array_equal(x=expected, y=split)

        # first index skipped
        # total group average is considered for each member regardless of index
        split = times.split(time=time, value=value, group_hours=3, step=3, region=(1, -1), whole_group=True)
        expected = [[2.5, 4.5, 6], [4.5, 6, 6], [4.5, 6, 6], [6, 6, 6]]
        np_test.assert_array_equal(x=expected, y=split)

        split = times.split(time=time, value=value, group_hours=3, step=-3, region=(4, 4), whole_group=True)
        expected = [[2.5, 4.5, 6]]
        np_test.assert_array_equal(x=expected, y=split)

    @staticmethod
    def test_one_hot():
        columns = ['01', '02', '03', '04']
        s = pd.to_datetime(pd.Series(data=[1, 2, 4]), utc=True, format='%H')
        one_hot = times.one_hot(times=s, columns=columns, time_format='%H')
        np_test.assert_array_equal(x=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], y=one_hot)

    def test_to_datetime(self):
        d_time = datetime.utcnow()
        d = d_time.date()
        expected = datetime.strptime(d_time.strftime('%y-%m-%d'), '%y-%m-%d')
        self.assertEqual(times.to_datetime(date=d), expected)
