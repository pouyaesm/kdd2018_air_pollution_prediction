import const
import pandas as pd


# Prepare time series data for learning
class Prepare:

    def __init__(self, config):
        self.config = config  # path of required files for preparing
        self.ts = pd.DataFrame()  # time series data
        self.stations = pd.DataFrame()  # stations data
        self.missing = pd.DataFrame()  # index of missing values in the data

    def load(self):
        """
            Load pre-processed data
        :return:
        """
        self.ts = pd.read_csv(self.config[const.OBSERVED], delimiter=';', low_memory=False)
        self.stations = pd.read_csv(self.config[const.STATIONS], delimiter=';', low_memory=False)
        self.missing = self.ts.isna()
        return self

    def save(self):
        """
            Save prepared data to files given in config
        :return:
        """
        print('Data saved.')


if __name__ == "__main__":
    prepare = Prepare({
        const.OBSERVED: const.BJ_OBSERVED,
        const.STATIONS: const.BJ_STATIONS
    }).load()

    print("Done!")


