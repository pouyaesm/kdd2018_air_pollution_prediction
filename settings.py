# setup.py
import configparser
import os

config = configparser.ConfigParser()
setup_path = os.path.dirname(os.path.realpath(__file__))
config.read(setup_path + '\\config.ini')

if __name__ == "__main__":
    print(config['DEFAULT']['AIRQUALITY'])

