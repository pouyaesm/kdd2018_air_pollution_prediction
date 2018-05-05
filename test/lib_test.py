import unittest
import pandas as pd
import pandas.util.testing as pd_test


class LibTest(unittest.TestCase):

    @staticmethod
    def test_dataframe_drop_duplicates():
        # expected to drop the second duplicate
        df_expected = pd.DataFrame(data={'id': [1], 'value': [1]})
        df_second = pd.DataFrame(data={'id': [1], 'value': [2]})
        df_dropped = df_expected.append(other=df_second).drop_duplicates(subset='id')
        pd_test.assert_frame_equal(left=df_expected, right=df_dropped)
