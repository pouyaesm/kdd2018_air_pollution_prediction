## Introduction

KDD 2018 held a competition to predict the intensity of air pollutants in Beijing and London for the next 48 hours during a month.
This is our team's code (`NIPL Rises`) that achieved 16th position (among 4000 teams) in [last 10 days category](https://biendata.com/competition/kdd_2018/ranking_last10/) executed on a single laptop.
We used `tensorflow` to build a hybrid model composed of CNNs, LSTMs, and MLPs for an end-to-end prediction over 2D grid data, time-series, and categorical data. This project includes (1) fetching and crawling of weather and pollutant data from multiple sources,
(2) data cleaning and integration, (3) visualization for insights,
and (4) prediction of pollutants (PM2.5, PM10, O3) for the next 48 hours in Beijing and London.

## Setup

1. Download csv files from [KDD 2018 data repository](https://biendata.com/competition/kdd_2018/data/) (requires sign-up),
1. Install required packages including `tensorflow` and `keras` (for deep learning), `selenium` (for web crawling),
1. Copy `default.config.ini` to `config.ini`,
1. Download `chromedriver.exe` for web crawling, and set the address
    ```
    CHROME_DRIVER_PATH = path to chromedriver.exe
    ```
1. Set the addresses of downloaded data-sets,
    ```
    BJ_AQ = Beijing air quality history
    BJ_AQ_REST = Beijing air quality history for 2nd and 3rd months of 2018
    BJ_AQ_STATIONS = Beijing air quality stations
    BJ_MEO = Beijing meteorological history
    BJ_GRID_DATA = Beijing grid weather history
    LD_* = same for London
    ```
1. Set the addresses for fetched/cleaned data to be stored,
    ```
    BJ_AQ_LIVE = fetched Beijing air quality live data
    BJ_MEO_LIVE = fetched Beijing meteorology live data
    BJ_OBSERVED = cleaned Beijing observed air quality and meteorology time series
    BJ_OBSERVED_MISS = marked missing data in BJ_OBSERVED
    BJ_STATIONS = cleaned data of stations in Beijing
    BJ_GRID_LIVE = fetched grid of current weather in Beijing
    BJ_GRID_FORECAST = fetched grid of forecast weather in Beijing
    BJ_GRIDS = history of grid data in Beijing
    BJ_GRID_COARSE = coarsened grid of data to lower resolutions
    LD_* = same for London
    ```
1. Set [lower, upper] bounds for date intervals of urls
    ```
    BJ_AQ_URL = */2018-06-05-0/2k0d1d8
    BJ_MEO_URL = */2018-06-05-0/2k0d1d8
    BJ_GRID_URL = */2018-06-05-0/2k0d1d8
    LD_*_URL = same for London
    ````
1. Set a path for generated features and models
    ```
    FEATURE_DIR = directory for extracted features
    MODEL_DIR = directory for generated models
    ```

## Execution

   1. Data pre-process
        1. Run `src/preprocess/preprocess_all.py` to create the cleaned data sets in your pre-specified addresses,
    1. Data visualization
        1. Run scripts in `src/statistics` to gain basic insights about value distributions, time series and geographical positions
        1. Change `BJ_*` to `LD_*` for London data
    1. Feature generation
        1. Go to main method of `src/feature_generators/hybrid_fg.py`,
        1. Uncomment desired (city-pollutant, sample rate) pairs in `cases` variable (all pairs are eventually required). Higher sample rate, larger data,
        1. Run the script
    1. Model training
        1. Go to `src/methods/lstm_pre_train.py`
        1. Run the script; simple LSTM models are pre-trained for all pollutants.
        These models are fed (unchanged) to the final model for better performance,
        1. Go to main method of `src/methods/hybrid.py`,
        1. Uncomment desired city-pollutant,
        1. Run the script; best model so far will be saved automatically
    1. Model testing
        1. Go to `src/methods/model_tests.py`,
        1. Uncomment desired city-pollutant, set a time interval in `TEST_FROM` and `TEST_TO`,
        1. Run the script; SMAPE score will be printed.
        1. Go to `src/methods/model_investigate.py`,
        1. Run the script to see SMAPE score per station sorted and geographically visualized.
    1. Prediction
        1. Go to `src/predict_next_48.py`
        1. Change `timedelta` if you wish to predict previous 48 hours,
        1. Run the script

## Toy Examples

1. `examples/gcforest` includes basic examples of using forests instead of neurons
to do deep learning proposed in [this paper](https://arxiv.org/abs/1702.08835),
2. `examples/tensorflow` includes basic examples of using tensorflow