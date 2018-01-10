#!/usr/bin/env python

import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from kg_recurit_restaurant_visitor_forecasting.src.commons import data_dir

logger_name = 'store_id_relation'
logging.getLogger(logger_name)
file_name = logger_name + '.csv'
file_path = data_dir + file_name


# air_visit_data.csv
# This file contains historical visit data for the air restaurants.
#
# air_store_id
# visit_date - the date
# visitors - the number of visitors to the restaurant on the date

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
