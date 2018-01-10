#!/usr/bin/env python

import logging

import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from kg_recurit_restaurant_visitor_forecasting.src.commons import data_dir
from datetime import timedelta

from kg_recurit_restaurant_visitor_forecasting.src.feature_handling.feature_generation_commons import gen_col_period

logger_name = 'air_reserve'
logging.getLogger(logger_name)
file_name = logger_name + '.csv'
file_path = data_dir + file_name


# air_reserve.csv
#
# This file contains reservations made in the air system. Note that the reserve_datetime indicates the time when the
# reservation was created, whereas the visit_datetime is the time in the future where the visit !will! occur.
# air_store_id - the restaurant's id in the air system
# visit_datetime - the time of the reservation
# reserve_datetime - the time the reservation was made
# reserve_visitors - the number of visitors for that reservation




def first_temp(file_path):
    ori_df = pd.read_csv(file_path)

    temp_df = ori_df.iloc[:10]

    df1 = temp_df.apply(gen_col_period, axis=1)

    temp = ori_df['reserve_visitors']
    temp.groupby(by=['reserve_visitors']).sum()
    # ori_df.shape
    # (92378, 4)

    # ori_df.columns
    # Index(['air_store_id', 'visit_datetime', 'reserve_datetime',
    #        'reserve_visitors'],
    #       dtype='object')

    ori_df.head(1)
    # air_store_id          visit_datetime          reserve_datetime        reserve_visitors
    # air_877f79706adbfb06  2016-01-01 19:00:00     2016-01-01 16:00:00     1

    ori_df['air_store_id'].drop_duplicates().shape
    # (314,)


def main():
    first_temp()


if __name__ == '__main__':
    main()
