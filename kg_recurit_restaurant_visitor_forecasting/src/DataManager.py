import os

import tensorflow as tf
import pandas as pd

COLUMNS_LIST = ['visit_date', 'nrv_0', 'nrv_1', 'nrv_2', 'nrv_3', 'nrv_4', 'nrv_5', 'nrv_6', 'nrv_7', 'nrv_8', 'nrv_9',
                'nrv_10', 'nrv_11', 'nrv_12', 'nrv_13', 'nrv_14', 'nrv_15', 'nrv_16', 'nrv_17', 'nrv_18', 'nrv_19',
                'nrv_20', 'nrv_21', 'nrv_22', 'nrv_23', 'nrv_24', 'nrv_25', 'nrv_26', 'nrv_27', 'nrv_28', 'nrv_29',
                'nrv_30', 'nrv_31', 'visitors', 'day_of_week', 'holiday_flg', 'air_genre_name', 'air_area_name',
                'latitude', 'longitude']

CHOOSEN_COLUMNS_LIST = ['nrv_0', 'nrv_1', 'nrv_2', 'nrv_3', 'nrv_4', 'nrv_5', 'nrv_6', 'nrv_7', 'nrv_8',
                        'nrv_9', 'nrv_10', 'nrv_11', 'nrv_12', 'nrv_13', 'nrv_14', 'nrv_15', 'nrv_16', 'nrv_17',
                        'nrv_18', 'nrv_19', 'nrv_20', 'nrv_21', 'nrv_22', 'nrv_23', 'nrv_24', 'nrv_25', 'nrv_26',
                        'nrv_27', 'nrv_28', 'nrv_29', 'nrv_30', 'nrv_31', 'visitors']


class DataManager(object):
    def __init__(self):
        self.features_file_dir = ''
        self.files_list = os.listdir(path=self.features_file_dir)
        self.file_index = 0
        self.files_num = len(self.files_list)
        self.batch_size = 5
        self.features_files_dir = '/home/bp/GitRepos/kg_recurit_restaurant_visitor_forecasting/kg_recurit_restaurant_visitor_forecasting/data/continue_date_store_files/'
        self.features_files_list = os.listdir(self.features_files_dir)
        pass

    def read_features(self, file_path):
        pass

    def list_files(self):
        features_files_dir = '/home/bp/GitRepos/kg_recurit_restaurant_visitor_forecasting/kg_recurit_restaurant_visitor_forecasting/data/continue_date_store_files/'
        features_files_list = os.listdir(features_files_dir)
        features_files_list

    def next_batch(self):
        index = 0
        data = []
        while (index < 5):
            goal_file = self.files_list[self.file_index + index]
            features = self.read_features(goal_file)
            data.append(features)
        self.file_index += 5
        pass


if __name__ == '__main__':
    file_path = '/home/bp/GitRepos/kg_recurit_restaurant_visitor_forecasting/kg_recurit_restaurant_visitor_forecasting/data/continue_date_store_files/air_0a74a5408a0b8642.csv'
    df = pd.read_csv(file_path)
    df = df[CHOOSEN_COLUMNS_LIST]
    values = df.values
    values
    features = values[:,:-1]
    labels = values[:,-1]
    features
    labels
    features.shape
    labels.shape
    temp_list = []
    temp_list.append(values)
    temp_list.append(values)
    temp_list
    import numpy as np
    temp = np.array(temp_list)
    temp
    temp.shape
