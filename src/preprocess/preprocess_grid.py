import settings
import const
import pandas as pd
from src import util
from src.preprocess.preprocess import PreProcess
from src.preprocess import times, reform


class PreProcessGrid(PreProcess):

    def __init__(self, config):
        super(PreProcessGrid, self).__init__(config)
        self.config = config  # location of input/output files
        self.obs = pd.DataFrame()  # merged observed air quality and meteorology data per station
        self.missing = pd.DataFrame()  # indicator of missing values in observed data-frame
        self.stations = pd.DataFrame()  # all stations with their attributes such as type and position

        self.sample_row = self.config[const.ROW]
        self.sample_column = self.config[const.COLUMN]
        self.row = 21
        self.city = self.config[const.CITY]
        self.column = 31 if self.city == const.BJ else 41
        self.block_row_size = util.ceildiv(self.row, self.sample_row)
        self.block_column_size = util.ceildiv(self.column, self.sample_column)

    def process(self):
        """
            Load and PreProcess the data
        :return:
        """
        iterators = dict()
        iterators['forecast'] = pd.read_csv(self.config[const.GRID_FORECAST], sep=';', iterator=True,
                                            low_memory=False, float_precision='round_trip')
        iterators['live'] = pd.read_csv(self.config[const.GRID_LIVE], sep=';', iterator=True,
                                        low_memory=False, float_precision='round_trip')
        iterators['history'] = (pd.read_csv(self.config[const.GRID_DATA], iterator=True, low_memory=False,
                               chunksize=1500000, float_precision='round_trip'))

        id_map = self.get_grid_id_maps()
        # id_grid is for visualizing purpose using data-frame viewer
        # id_list = [id_map[grid_id] for grid_id in sorted(id_map.keys())]
        # id_grid = np.flipud(np.array(id_list).reshape((self.row, self.column), order='F'))
        collection = {measure: dict() for measure in self.get_measures()}

        # Add historical / live / forecast grid data to coarsened statistics
        for category, grid in iterators.items():
            for i, chunk in enumerate(grid):
                print(' merge grid chunk (%s, %d) ..' % (category, i + 1))
                chunk.rename(columns={'stationName': const.GID, 'wind_speed/kph': const.WSPD}, inplace=True)
                # convert wind speed and direction to polar values (x, y)
                chunk[const.WSPD], chunk[const.WDIR] = reform.wind_transform(
                    speed=chunk[const.WSPD], direction=chunk[const.WDIR])
                time_group = chunk.groupby([const.TIME])  # each time is a square of points
                for time, group in time_group:
                    self.add(time, group, id_map, collection)

        # Add live grid data
        for measure in collection:
            collect = collection[measure]
            # build final data table sorted by time ascending
            data = list()
            for time, stats in sorted(collect.items()):
                values = [v / c if c > 0 else 0 for v, c in
                           zip(stats['values'], stats['counts'])]
                data.append([time] + values)
            columns = self.get_columns()
            df = pd.DataFrame(data=data, columns=columns)
            df[const.TIME] = pd.to_datetime(df[const.TIME], utc=True)
            # group_hours = self.config[const.GROUP_HOURS]
            # run_average_df = times.running_average_df(df=df, time_key=const.TIME, value_keys=columns[1:],
            #                                           group_hours=group_hours, direction=1, whole_group=False)
            print('%d x (%d, %d) coarsened grid generated for (%s, %s)' % (
                len(collect), self.sample_row, self.sample_column, self.city, measure))
            util.write(df, self.config[const.GRID_COARSE] % measure)

        return self

    def add(self, time, time_group, id_map, collection):
        columns = {column: index + 1 for index, column in enumerate(time_group.columns)}
        for measure, collect in collection.items():
            measure_index = columns[measure]
            id_index = columns[const.GID]
            # if a time is separated between two loaded chunks continue the existing statistics
            if time in collect:
                values = collect[time]['values']
                counts = collect[time]['counts']
            else:
                values = [0] * self.sample_row * self.sample_column
                counts = [0] * len(values)
            for grid in time_group.itertuples():
                sample_id = id_map[grid[id_index]]
                values[sample_id] += grid[measure_index]
                counts[sample_id] += 1
            collect[time] = {'values': values, 'counts': counts}

    @staticmethod
    def get_measures():
        return [const.TEMP, const.HUM, const.WSPD, const.WDIR]

    def get_columns(self):
        columns = [const.TIME]
        count = self.sample_row * self.sample_column
        columns.extend(['v%d' % c for c in range(0, count)])
        return columns

    def get_grid_id_maps(self):
        """
            Create a map between grid-ids and their index in the sub-sampled vector presentation
            Ids are incremented to the north then reset from west to east
        :return:
        """
        # convert grid indices to indices after sub-sampling
        # and changing the id increment from (to north, to east) to (to east, to south)
        prefix = 'beijing' if self.city == const.BJ else 'london'
        id2index = {('%s_grid_%03d' % (prefix, i)): self.to_row(i) * self.sample_column + self.to_col(i)
                    for i in range(0, self.row * self.column)}
        return id2index

    # convert general grid id to sub-sampled grid id with change of increment direction
    def to_row(self, i):
        return (self.row - (i % self.row) - 1) // self.block_row_size

    def to_col(self, i):
        return (i // self.row) // self.block_column_size


if __name__ == "__main__":
    config = settings.config[const.DEFAULT]
    cities = [const.BJ, const.LD]
    for city in cities:
        cfg = {
            const.CITY: city,
            const.GRIDS: config[getattr(const, city + '_GRIDS')],
            const.GRID_DATA: config[getattr(const, city + '_GRID_DATA')],
            const.GRID_LIVE: config[getattr(const, city + '_GRID_LIVE')],
            const.GRID_FORECAST: config[getattr(const, city + '_GRID_FORECAST')],
            const.GRID_COARSE: config[getattr(const, city + '_GRID_COARSE')],
            const.ROW: 5,
            const.COLUMN: 5,
        }
        pre_process = PreProcessGrid(cfg)
        pre_process.process()
    print("Done!")
