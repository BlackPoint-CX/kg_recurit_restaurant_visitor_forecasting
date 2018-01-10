#!/usr/bin/env python

import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from kg_recurit_restaurant_visitor_forecasting.src.commons import data_dir

logger_name = 'date_info'
logging.getLogger(logger_name)
file_name = logger_name + '.csv'
file_path = data_dir + file_name


# date_info.csv
# This file gives basic information about the calendar dates in the dataset.
#
# calendar_date
# day_of_week
# holiday_flg - is the day a holiday in Japan

def first_temp(file_path):
    ori_df = pd.read_csv(file_path)

    ori_df.shape
    #

    ori_df.columns
    #

    ori_df.head(1)
    #

    ori_df[''].drop_duplicates().shape
    #


def main():
    first_temp()


if __name__ == '__main__':
    main()
