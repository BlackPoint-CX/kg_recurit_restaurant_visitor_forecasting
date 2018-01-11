#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# Author(s) : BlackPoint-CX
# CreateTime : 2018/1/10 18:36

import os

import numpy as np
import pandas as pd

CONTINUE_FILES_PATH = ''
BATCH_SIZE = 5


def gen_features(file_path):
    """
    读取单个文件, 并将train和predict部分分割.
    :param file_path:
    :return:
    """
    file_ori_df = pd.read_csv(file_path)
    train_df = np.array(file_ori_df[:480].values)
    train_features = train_df[:-1]
    train_labels = train_df[-1:]
    predict_df = np.array(file_ori_df[480:].values)
    predict_features = predict_df[:-1]
    predict_labels = predict_df[-1:]
    return train_features, train_labels, predict_features, predict_labels


def raw_data():
    """
    从目标文件夹下每次读取batch_size个文件, 并将文件中的数据通过gen_features进行features,labels划分
    :return:
    """
    files_list = os.listdir(CONTINUE_FILES_PATH)
    batch_num = len(files_list) // BATCH_SIZE

    all_train_features = []
    all_train_labels = []
    all_predict_features = []
    all_predict_labels = []
    for i in range(0, batch_num):
        batch_train_features = []
        batch_train_labels = []
        batch_predict_features = []
        batch_predict_labels = []
        sub_files_list = files_list[i * batch_num: (i + 1) * batch_num]
        for file_name in sub_files_list:
            file_path = CONTINUE_FILES_PATH + file_name
            train_features, train_labels, predict_features, predict_labels = gen_features(file_path=file_path)
            batch_train_features.append(train_features)
            batch_train_labels.append(train_labels)
            batch_predict_features.append(predict_features)
            batch_predict_labels.append(predict_labels)
        all_train_features.append(batch_train_features)
        all_train_labels.append(batch_train_labels)
        all_predict_features.append(batch_predict_features)
        all_predict_labels.append(batch_predict_labels)
    # 转换为np.array
    all_train_features = np.array(all_train_features)
    all_train_labels = np.array(all_train_labels)
    all_predict_features = np.array(all_predict_features)
    all_predict_labels = np.array(all_predict_labels)
    return all_train_features, all_train_labels, all_predict_features, all_predict_labels
