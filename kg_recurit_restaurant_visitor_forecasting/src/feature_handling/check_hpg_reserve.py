#!/usr/bin/env python

import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from kg_recurit_restaurant_visitor_forecasting.src.commons import data_dir

logger_name = 'hpg_reserve'
logging.getLogger(logger_name)
file_name = logger_name + '.csv'
file_path = data_dir + file_name


# hpg_reserve.csv
# This file contains reservations made in the hpg system.
#
# hpg_store_id - the restaurant's id in the hpg system
# visit_datetime - the time of the reservation
# reserve_datetime - the time the reservation was made
# reserve_visitors - the number of visitors for that reservation


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
