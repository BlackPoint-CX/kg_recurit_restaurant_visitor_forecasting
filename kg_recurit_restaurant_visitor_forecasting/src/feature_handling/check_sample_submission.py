#!/usr/bin/env python

import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from kg_recurit_restaurant_visitor_forecasting.src.commons import data_dir

logger_name = 'sample_submission'
logging.getLogger(logger_name)
file_name = logger_name + '.csv'
file_path = data_dir + file_name


# sample_submission.csv
# This file shows a submission in the correct format, including the days for which you must forecast.
#
# id - the id is formed by concatenating the air_store_id and visit_date with an underscore
# visitors- the number of visitors forecasted for the store and date combination

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
