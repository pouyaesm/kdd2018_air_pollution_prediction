# coding=utf-8
# Real-time air quality data from Beijing official environmental department: http://zx.bjmemc.com.cn/getAqiList.shtml
import datetime
import pandas as pd
from selenium import webdriver
import requests
import const
import settings

config = settings.config[const.DEFAULT]


def get_beijing_aq_latest():
    station_id_mapping = {
        1: "yongdingmennei_aq",  # 永定门
        2: "yufa_aq",  # 京南
        3: "zhiwuyuan_aq",  # 香山
        5: "fengtaihuayuan_aq",  # 丰台花园
        6: "shunyi_aq",  # 顺义新城
        7: "yanqin_aq",  # 夏都
        8: "pinggu_aq",  # 平谷镇
        9: "fangshan_aq",  # 良乡
        10: "yizhuang_aq",  # 亦庄
        11: "yungang_aq",  # 云岗
        12: "miyunshuiku_aq",  # 京东北
        13: "huairou_aq",  # 怀柔镇
        14: "badaling_aq",  # 京西北
        15: "wanshouxigong_aq",  # 万寿西宫
        17: "pingchang_aq",  # 昌平镇
        18: "mentougou_aq",  # 双峪
        19: "tongzhou_aq",  # 通州北苑
        20: "daxing_aq",  # 黄村
        21: "dingling_aq",  # 定陵
        23: "qianmen_aq",  # 前门
        24: "dongsi_aq",  # 东四
        25: "tiantan_aq",  # 天坛
        26: "aotizhongxin_aq",  # 奥体中心
        27: "nongzhanguan_aq",  # 农展馆
        28: "miyun_aq",  # 密云镇
        29: "gucheng_aq",  # 古城
        32: "guanyuan_aq",  # 西城官园
        34: "nansanhuan_aq",  # 南三环
        37: "beibuxinqu_aq",  # 北部新区
        38: "wanliu_aq",  # 海淀万柳
        40: "yongledian_aq",  # 京东南
        41: "liulihe_aq",  # 京西南
        43: "donggaocun_aq",  # 京东
        46: "dongsihuan_aq",  # 东四环
        47: "xizhimenbei_aq",  # 西直门
    }
    url = "http://zx.bjmemc.com.cn/getAqiList.shtml"
    driver = webdriver.Chrome(executable_path=config[const.CHROME_DRIVER_PATH])
    driver.get(url)
    data = driver.execute_script("return wfelkfbnx;")
    rows = list()
    for item in data:
        _id = item["id"]
        if _id not in station_id_mapping:
            continue
        station_id = station_id_mapping[_id]
        pm2_5 = item["pm2_01"]
        pm10 = item["pm10_01"]
        no2 = item["no2_01"]
        co = item["co_01"]
        o3 = item["o3_01"]
        so2 = item["so2_01"]
        time_str = (datetime.datetime.utcfromtimestamp(int(item["date_f"])) - datetime.timedelta(hours=8)).strftime(
            '%Y-%m-%d %H:%M:%S')
        rows.append([station_id, time_str, pm2_5, pm10, o3, no2, co, so2])
    driver.close()
    df = pd.DataFrame(data=rows, columns=[const.ID, const.TIME, const.PM25, const.PM10,
                                             const.O3, const.NO2, const.CO, const.SO2])
    return df


def get_beijing_historical(chunk_size = 1):
    start_date = datetime.date(2015, 1, 1)
    end_date = datetime.date(2017, 1, 1)
    date_range = pd.date_range(start_date, end_date)
    for i in date_range:
        # PM2.5/PM10/AQI
        path = ("http://beijingair.sinaapp.com/data/beijing/all/%s/csv" % i.strftime('%Y%m%d'))
        res = requests.get(path)
        file_path = ("data/%s.csv" % i.strftime('%Y%m%d'))
        with open(file_path, 'w+') as csvfile:
            csvfile.write(res.text)
    for i in pd.date_range(start_date, end_date):
        # SO2/NO2/O3/CO
        path = ("http://beijingair.sinaapp.com/data/beijing/extra/%s/csv" % i.strftime('%Y%m%d'))
        res = requests.get(path)
        file_path = ("./Data/extra/%s.csv" % i.strftime('%Y%m%d'))
        with open(file_path, 'w+') as csvfile:
            csvfile.write(res.text)
