import unittest
import pandas as pd
import const
from src.preprocess.preprocess_grid import PreProcessGrid
import pandas.util.testing as pd_test
import numpy.testing as np_test


class PreProcessGridTest(unittest.TestCase):

    def test_id_maps(self):
        pre_process = PreProcessGrid({const.CITY: const.BJ, const.ROW: 5, const.COLUMN: 5})
        id_maps = pre_process.get_grid_id_maps()
        self.assertEqual(first=20, second=id_maps['beijing_grid_000'])
        self.assertEqual(first=0, second=id_maps['beijing_grid_020'])
        self.assertEqual(first=4, second=id_maps['beijing_grid_650'])

    def test_add(self):
        # addition of a time group to previous collection based on id map
        pre_process = PreProcessGrid({const.CITY: const.BJ, const.ROW: 10, const.COLUMN: 10})
        id_map = {'a1': 0, 'a2': 2}
        collection = {'c1': dict(), 'c2': dict()}
        time = '2017-01-01'
        group_1 = pd.DataFrame(data={const.GID: ['a1', 'a2'], 'c1': [1, 2], 'c2': [10, 20]})
        group_2 = pd.DataFrame(data={const.GID: ['a1', 'a2'], 'c1': [3, 4], 'c2': [30, 40]})
        pre_process.add(time, group_1, id_map, collection)
        pre_process.add(time, group_2, id_map, collection)
        # Expect c# values of a1 summed in [0] and c# values of a2 summed in [2]
        a1_values = collection['c2'][time]['values'][0]
        a2_values = collection['c2'][time]['values'][2]
        a1_counts = collection['c2'][time]['counts'][0]
        self.assertEqual(first=40, second=a1_values)
        self.assertEqual(first=60, second=a2_values)
        self.assertEqual(first=2, second=a1_counts)
