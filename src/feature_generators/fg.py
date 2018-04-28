import pandas as pd
import const


class FG:

    @staticmethod
    def normalize_df(df: pd.DataFrame, names: list, city):
        for name in names:
            df[name] = FG.normalize(df[name], name=name, city=city)
        return df

    @staticmethod
    def de_normalize_df(df: pd.DataFrame, names: list, city):
        for name in names:
            df[name] = FG.de_normalize(df[name], name=name, city=city)
        return df

    @staticmethod
    def normalize(values: pd.Series, name, city):
        mean, std = FG.get_statistics(name=name, city=city)
        normal = (values - mean) / std
        return normal

    @staticmethod
    def de_normalize(values: pd.Series, name, city):
        mean, std = FG.get_statistics(name=name, city=city)
        de_normal = (values * std) + mean
        return de_normal

    @staticmethod
    def get_statistics(name, city):
        """
        :param name:
        :return: (mean, std)
        :rtype: tuple (float, float)
        """
        if city == const.BJ:
            if name == const.PM25:
                return 61, 67
            elif name == const.PM10:
                return 92, 100
            elif name == const.O3:
                return 56, 52
            elif name == const.TEMP:
                return 11, 12
            elif name == const.PRES:
                return 1002, 22
            elif name == const.HUM:
                return 42, 23
            elif name == const.WDIR:
                return 180, 107
            elif name == const.WSPD:
                return 6.3, 5.9
        elif city == const.LD:
            if name == const.PM25:
                return 13.3, 11
            elif name == const.PM10:
                return 20, 14
            elif name == const.TEMP:
                return 10.3, 5.9
            elif name == const.PRES:
                return 1010, 11
            elif name == const.HUM:
                return 77, 13
            elif name == const.WSPD:
                return 15.4, 7.3
            elif name == const.WDIR:
                return 212, 88
