#!/usr/bin/env python

import logging

import datetime
import matplotlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from kg_recurit_restaurant_visitor_forecasting.src.commons import data_dir, chart_dir, set_logger
from kg_recurit_restaurant_visitor_forecasting.src.feature_handling.feature_generation_commons import gen_col_period, \
    gen_col_period_dt

logger_name = 'store_id_relation'
logger = set_logger(logger_name)

data_dir = '/home/bp/GitRepos/kg_recurit_restaurant_visitor_forecasting/kg_recurit_restaurant_visitor_forecasting/data/'

air_reserve_path = data_dir + 'air_reserve.csv'
air_store_info_path = data_dir + 'air_store_info.csv'
air_visit_info_path = data_dir + 'air_visit_info.csv'
hpg_reserve_path = data_dir + 'hpg_reserve.csv'
hpg_store_info_path = data_dir + 'hpg_store_info.csv'
date_info_path = data_dir + 'date_info.csv'
store_id_relation_path = data_dir + 'store_id_relation.csv'

data_features_dir = data_dir + 'features/'
single_store_files_dir = data_dir + 'single_store_files/'
continue_store_files_dir = data_dir + 'continue_date_store_files/'


def features_gen_step_0():
    store_id_relation_df = pd.read_csv(data_dir + 'store_id_relation.csv')

    air_reserve_df = pd.read_csv(air_reserve_path)
    # air_reserve_df_2 = air_reserve_df.apply(func=gen_col_period, axis=1)
    # air_reserve_df_2.to_csv(data_features_dir + 'air_reserve_df_2.csv')
    # ar_df_2_rv = air_reserve_df_2['reserve_visitors'].value_counts().sort_index()
    # fig = ar_df_2_rv.plot(kind='bar')
    # fig.set_title('air_reserve reserve_visitor distribution')
    # plt.savefig(chart_dir + 'air_reserve|reserve_visitor|dist')

    hpg_reserve_df = pd.read_csv(hpg_reserve_path)
    # hpg_reserve_df_2 = hpg_reserve_df.apply(gen_col_period, axis=1)
    # hpg_reserve_df_2.to_csv(data_features_dir + 'hpg_reserve_df_2.csv')
    # hpg_df_r_rv = hpg_reserve_df_2['reserve_visitors'].value_counts().sort_index()
    # fig = hpg_df_r_rv.plot(kind='bar')
    # fig.set_title('hpg_reserve reserve_visitor distribution')
    # plt.savefig(chart_dir + 'hpg_reserve|reserve_visitor|dist')

    air_reserve_union_hpg = pd.merge(left=air_reserve_df, right=store_id_relation_df, how='outer',
                                     left_on='air_store_id', right_on='air_store_id')
    hpg_reserve_union_air = pd.merge(left=hpg_reserve_df, right=store_id_relation_df, how='outer',
                                     left_on='hpg_store_id', right_on='hpg_store_id')

    air_hpg_all_df = pd.concat([air_reserve_union_hpg, hpg_reserve_union_air])
    air_hpg_all_df.to_csv(data_features_dir + 'air_hpg_all.csv')

    # 因为air_reserve_union_hpg有部分记录是合并时outer进来的只有hpg_restore_id没有air_store_id的记录, 所以需要清洗.
    air_hpg_all_clean_df = air_hpg_all_df[
        (air_hpg_all_df['reserve_datetime'].notnull()) | (air_hpg_all_df['visit_datetime'].notnull())]  # (2092698, 5)

    # 将对应的列进行简单的格式变换和解析
    air_hpg_all_clean_sim_formatted = air_hpg_all_clean_df.apply(gen_col_period, axis=1)
    air_hpg_all_clean_sim_formatted.to_csv(data_features_dir + 'air_hpg_all_clean_sim_formatted.csv', index=False)

    # air_hpg_all_clean_sim_formatted = pd.read_csv(data_features_dir + 'air_hpg_all_clean_sim_formatted.csv')

    # air_hpg_all_clean_sim_formatted[air_hpg_all_clean_sim_formatted['air_store_id'].isnull()]['hpg_store_id'].drop_duplicates()
    # notnull 151 isnull 13175

    # 找出所有air_store_id非空的记录
    air_hpg_all_clean_sim_formatted_air_id_notnull = air_hpg_all_clean_sim_formatted[
        air_hpg_all_clean_sim_formatted['air_store_id'].notnull()]
    air_hpg_all_clean_sim_formatted_air_id_notnull_grouped = air_hpg_all_clean_sim_formatted_air_id_notnull.groupby(
        by='air_store_id')

    air_store_id_list = air_hpg_all_clean_sim_formatted_air_id_notnull['air_store_id'].drop_duplicates().values.tolist()
    for air_store_id in air_store_id_list:
        print('Processing : %s' % air_store_id)
        air_store_single = air_hpg_all_clean_sim_formatted_air_id_notnull_grouped.get_group((air_store_id))
        air_store_single.to_csv(single_store_files_dir + air_store_id + '.csv', index=False)


def features_gen_step_2():
    single_store_files_dir = '/home/bp/GitRepos/kg_recurit_restaurant_visitor_forecasting/kg_recurit_restaurant_visitor_forecasting/data/single_store_files/'
    air_visit_df = pd.read_csv(data_dir + 'air_visit_data.csv', index_col='air_store_id')
    air_store_info_df = pd.read_csv(data_dir + 'air_store_info.csv', index_col='air_store_id')
    date_info_df = pd.read_csv(data_dir + 'date_info.csv', index_col='calendar_date')
    files_list = os.listdir(single_store_files_dir)
    print('Start(features_gen_step_2) ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    for file_name in files_list:
        file_path = single_store_files_dir + file_name
        air_store_id = file_name.split(sep='.')[0]  # air_store_id = 'air_0a74a5408a0b8642'
        print('Processing File : %s' % air_store_id)
        visit_df = air_visit_df.loc[air_store_id]
        store_df = air_store_info_df.loc[air_store_id]
        single_store_file_handle(file_name, file_path, visit_df, store_df, date_info_df)
    print('End(features_gen_step_2) ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def features_gen_step_3():
    for file_name in os.listdir(continue_store_files_dir):
        file_df = pd.read_csv(continue_store_files_dir + file_name)


def single_store_file_handle(single_store_file_name, single_store_file_path, visit_df, store_df, date_info_df):
    # dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    # single_store_df = pd.read_csv(single_store_file_path,index_col='visit_date',parse_dates=['reserve_date','visit_date'],date_parser=dateparse)
    # single_store_file_path = '/home/bp/GitRepos/kg_recurit_restaurant_visitor_forecasting/kg_recurit_restaurant_visitor_forecasting/data/single_store_files/air_0a74a5408a0b8642.csv'
    single_store_df = pd.read_csv(single_store_file_path)
    # Index(['air_store_id', 'hpg_store_id', 'reserve_datetime', 'reserve_visitors', 'visit_datetime', 'reserve_date',
    # 'reserve_hour', 'visit_date', 'visit_hour', 'period_days', 'period_hours'], dtype='object')
    single_store_df.drop(['visit_datetime', 'reserve_datetime', 'air_store_id', 'hpg_store_id'], axis=1, inplace=True)
    # Index(['reserve_visitors', 'reserve_date', 'reserve_hour', 'visit_date','visit_hour', 'period_days', 'period_hours'],dtype='object')
    grouped = single_store_df.groupby(by=['visit_date', 'period_days'])
    # grouped.get_group(('2016-10-31', 0))
    group1 = grouped['reserve_visitors'].sum()
    group2 = group1.reset_index()
    group2.set_index(group2['visit_date'], inplace=True)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 根据历史数据,在当前日期之前31填预约到今天的预约数, 并分别入列.
    # 例如, 在 '2017-04-16', 之前分别在 0,3,4,5,10 天之前预约了 2,5,14,4,2 个位置, 那么就应该分别在对应位置填入正确的预约数
    # example_date = '2017-04-16'
    # group2.loc[example_date]
    # final_df.loc[example_date]
    columns_list = []
    for i in range(0, 32, 1):
        col_name = 'nrv_%d' % i
        columns_list.append(col_name)

    final_df = pd.DataFrame(columns=columns_list)
    final_df['visit_date'] = group2['visit_date']
    final_df.set_index(final_df['visit_date'], inplace=True)
    # number of reserve visitors last x days ago
    # 已经优化过了, 但是仍不满意, 会产生duplicates, 需要在最后再做一次drop_duplicates操作, 不明白为什么会产生重复值.
    for i in range(0, 32, 1):
        # print('processing : %d' % i)
        col_name = 'nrv_%d' % i
        final_df.loc[final_df['visit_date'], col_name] = \
            group2[(group2['visit_date'] == final_df['visit_date']) & (group2['period_days'] == i)]['reserve_visitors']
    final_df.drop_duplicates(inplace=True)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 填充不足的日期
    period_start = final_df.index.min()
    period_end = final_df.index.max()
    index_all_date = pd.date_range(start=period_start, end=period_end, freq='D')
    index_all_df = pd.DataFrame(index=index_all_date)
    temp1 = pd.merge(index_all_df, final_df, left_index=True, right_index=True, how='outer')
    temp1.fillna(value=0.0, inplace=True)
    temp1.drop(['visit_date'], axis=1, inplace=True)

    # 和visit_data进行合并date_info_df
    visit_df = visit_df.reset_index().drop(['air_store_id'], axis=1)
    visit_df.set_index('visit_date', inplace=True)
    union_df = pd.merge(left=temp1, right=visit_df, how='outer', left_index=True, right_index=True)

    # 和date_info进行合并
    union_df = pd.merge(left=union_df, right=date_info_df, how='outer', left_index=True, right_index=True)

    # 和store_info进行合并
    union_df['air_genre_name'] = store_df['air_genre_name']
    union_df['air_area_name'] = store_df['air_area_name']
    union_df['latitude'] = store_df['latitude']
    union_df['longitude'] = store_df['longitude']

    # 存储
    union_df.index.name = 'visit_date'
    union_df = union_df.fillna(value=0.0)
    union_df.to_csv(continue_store_files_dir + single_store_file_name)


def specific_air_store_info():
    air_store_info_df = pd.read_csv(air_store_info_path)
    air_store_info_df_splitted = air_store_info_df.apply(split_air_area_name,axis=1)
    air_store_info_df_splitted.to_csv(data_features_dir + 'air_store_info_splitted.csv',index=False)

def specific_hpg_store_info():
    hpg_store_info_df = pd.read_csv(hpg_store_info_path)
    hpg_store_info_df_splitted = hpg_store_info_df.apply(split_hpg_area_name,axis=1)
    hpg_store_info_df_splitted.to_csv(data_features_dir + 'hpg_store_info_splitted.csv',index=False)

def split_air_area_name(row):
    air_area_name = row['air_area_name']
    air_area_name_list = air_area_name.split(' ')
    len_air_area_name_list = len(air_area_name_list)
    idx = 0
    for i in range(0, 3, 1):
        col_name = 'air_sub_area_%d' % i
        row[col_name] = 'EmptySubArea'
        while idx < len_air_area_name_list:
            if air_area_name_list[idx].isdigit():
                idx += 1
                continue
            else:
                row[col_name] = air_area_name_list[idx]
                idx += 1
                break
    return row

def split_hpg_area_name(row):
    hpg_area_name = row['hpg_area_name']
    hpg_area_name_list = hpg_area_name.split(' ')
    len_hpg_area_name_list = len(hpg_area_name_list)
    idx = 0
    for i in range(0, 3, 1):
        col_name = 'hpg_sub_area_%d' % i
        row[col_name] = 'EmptySubArea'
        while idx < len_hpg_area_name_list:
            if hpg_area_name_list[idx].isdigit():
                idx += 1
                continue
            else:
                row[col_name] = hpg_area_name_list[idx]
                idx += 1
                break
    return row

def features_gen_step_4():
    data_dir = '/home/bp/GitRepos/kg_recurit_restaurant_visitor_forecasting/kg_recurit_restaurant_visitor_forecasting/data/'
    date_info_df = pd.read_csv(data_dir + 'date_info.csv')
    date_info_df.columns
    temp = pd.get_dummies(data=date_info_df, columns=['day_of_week', 'holiday_flg'])
    temp
    type(temp)
    temp.columns

    air_store_info = pd.read_csv(data_dir + 'air_store_info.csv')
    air_store_info.columns
    air_store_info.head(1)
    temp = pd.get_dummies(data=air_store_info, columns=['air_genre_name', 'air_area_name'])
    temp
    type(temp)
    temp.columns


def main():
    pass


if __name__ == '__main__':
    main()
