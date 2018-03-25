## Introduction

A tool for analysing and predicting air and weather data from KDD 2018 competition

## Setup

1. Copy `default.config.ini` to `config.ini`,
2. Set addresses for air quality and mereological (weather) data-sets,
3. Set the address for station data-set,
4. Set an address for the output of pre-processing (cleaned data)

## Execution

1. Run `src/preprocess.py` to create the cleaned data set,
2. Run scripts in `src/statistics` for gaining basic insights

## Examples

1. `examples/gcforest` includes basic examples of using forests instead of neurons
to do deep learning proposed in [this paper](https://arxiv.org/abs/1702.08835),
2. `examples/tensorflow` includes basic examples of using tensorflow

## On-going jobs

1. Add a training example for time series to `examples/tensorflow`