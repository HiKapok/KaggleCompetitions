
# coding: utf-8

# In[ ]:


# The line below sets the environment
# variable CUDA_VISIBLE_DEVICES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp      # will come in handy due to the size of the data
import os.path
import random
import time
from collections import OrderedDict
import io
from datetime import datetime
import gc # garbage collector
import sklearn
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import math
import sys
from collections import defaultdict
import re
import logging
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search


# ## Write a pandas dataframe to disk as gunzip compressed csv
# - df.to_csv('dfsavename.csv.gz', compression='gzip')
# 
# ## Read from disk
# - df = pd.read_csv('dfsavename.csv.gz', compression='gzip')
# 
# ## Magic useful
# - %%timeit for the whole cell
# - %timeit for the specific line
# - %%latex to render the cell as a block of latex
# - %prun and %%prun

# In[ ]:


DATASET_PATH = '/media/rs/0E06CD1706CD0127/Kapok/WSDM/'
HDF_FILENAME = DATASET_PATH + 'datas.h5'
HDF_FILENAME_TEMPSAVE = DATASET_PATH + 'datas_temp.h5'
SUBMISSION_FILENAME = DATASET_PATH + 'submission_{}.csv'
VALIDATION_INDICE = DATASET_PATH + 'validation_indice.csv'


# In[ ]:


def set_logging(logger_name, logger_file_name):
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    print_formatter = logging.Formatter('%(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s_%(levelname)s: %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler(logger_file_name, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)
    log.addHandler(fh)
    # both output to console and file
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(print_formatter)
    log.addHandler(consoleHandler)
    
    return log


# In[ ]:


log = set_logging('MUSIC', DATASET_PATH + 'music_gbm.log')
log.info('here is an info message.')


# In[ ]:


store_test = pd.HDFStore(HDF_FILENAME)
test_id =  store_test['test_id']
store_test.close()


# In[ ]:


# del train_use, validation_use, test
gc.collect()
float32_list = ['genre_ids', 'language', 'song_year', 'composer_score', 'lyricist_score', 'artist_name_score', 'popular_0', 'num_people_0', 'popular_1', 'popular_2', 'popular_3', 'popular_4', 'active_0', 'num_song_0', 'active_1', 'active_2', 'active_3', 'active_4', 'song_id_by_city', 'msno_by_country', 'msno_by_language', 'genre_ids_score', 'msno_is_in_topK_of_artist_name', 'msno_is_in_topK_of_genre_ids', 'artist_name_is_in_topK_of_genre_ids', 'artist_name_is_in_topK_of_msno', 'source_screen_name_count', 'source_type_count', 'source_system_tab_count', 'source_screen_name_avg_score', 'source_type_avg_score', 'source_system_tab_avg_score', 'composer_by_city_country_language', 'lyricist_by_city_country_language', 'artist_name_by_city_country_language', 'city_hot', 'language_hot']
data_type_map =dict(zip(float32_list, [np.float32]*len(float32_list))) 
train_use = pd.read_csv(DATASET_PATH + 'temp_train_all_comp_encode_id.csv', compression='gzip', dtype = data_type_map)
gc.collect()
validation_use = pd.read_csv(DATASET_PATH + 'temp_validation_all_comp_encode_id.csv', compression='gzip', dtype = data_type_map)
test = pd.read_csv(DATASET_PATH + 'temp_test_all_comp_encode_id.csv', compression='gzip', dtype = data_type_map)
#test_id =  test['id']
gc.collect()


# In[ ]:


def catogory_encode_transform(train_data, val_data, test_data, col):
    gc.collect()
    temp_data = pd.concat([train_data, val_data, test_data], axis=0, join="outer")
    gc.collect()
    all_values = list(temp_data[col].unique())
    gc.collect()
    map_dict = dict(zip(all_values, [i for i in range(len(all_values))]))
    gc.collect()
#     train_data[col] = train_data[col].map(map_dict)
#     val_data[col] = val_data[col].map(map_dict)
#     test_data[col] = test_data[col].map(map_dict)
    return train_data[col].map(map_dict), val_data[col].map(map_dict), test_data[col].map(map_dict)


# In[ ]:


# col = 'msno'
# train_use[[col]], validation_use[[col]], test[[col]] = catogory_encode_transform(train_use[[col]], validation_use[[col]], test[[col]], col)
# gc.collect()
# col = 'song_id'
# train_use[[col]], validation_use[[col]], test[[col]] = catogory_encode_transform(train_use[[col]], validation_use[[col]], test[[col]], col)
# gc.collect()
# col = 'name'
# train_use[[col]], validation_use[[col]], test[[col]] = catogory_encode_transform(train_use[[col]], validation_use[[col]], test[[col]], col)
# gc.collect()
# col = 'composer'
# train_use[[col]], validation_use[[col]], test[[col]] = catogory_encode_transform(train_use[[col]], validation_use[[col]], test[[col]], col)
# gc.collect()
# col = 'lyricist'
# train_use[[col]], validation_use[[col]], test[[col]] = catogory_encode_transform(train_use[[col]], validation_use[[col]], test[[col]], col)
# gc.collect()
# col = 'artist_name'
# train_use[[col]], validation_use[[col]], test[[col]] = catogory_encode_transform(train_use[[col]], validation_use[[col]], test[[col]], col)
# gc.collect()


# In[ ]:


for col in test.columns:
    if col not in ['song_length', 'id']:
        if test[col].dtype == np.float64:
            train_use[col] = train_use[col].astype(np.float32)
            validation_use[col] = validation_use[col].astype(np.float32)
            test[col] = test[col].astype(np.float32)


# In[ ]:


# for col in ['msno', 'song_id', 'name', 'composer', 'lyricist', 'artist_name']:
#     train_use[col] = train_use[col].astype('category')
#     validation_use[col] = validation_use[col].astype('category')
#     test[col] = test[col].astype('category')


# In[ ]:


for col in [col for col in test.columns if col != 'id' ]:
    if train_use[col].dtype == object:
        train_use[col] = train_use[col].astype('category')
        validation_use[col] = validation_use[col].astype('category')
        test[col] = test[col].astype('category')


# In[ ]:


for col in test.columns:
    if col in ['registered_via', 'bd', 'city', 'registration_year', 'registration_month', 'registration_day', 'expiration_year', 'expiration_year', 'expiration_day']:
        if test[col].dtype == np.int64:
            train_use[col] = train_use[col].astype(np.int32)
            validation_use[col] = validation_use[col].astype(np.int32)
            test[col] = test[col].astype(np.int32)


# In[ ]:


# float32_list = list()
# for col in test.columns:
#     if col not in ['song_length', 'id']:
#         if test[col].dtype == np.float32:
#             float32_list.append(col)
# print(float32_list)


# In[ ]:


# train_use.to_csv(DATASET_PATH + 'temp_train_all_comp_encode_id.csv', index = False, compression='gzip')
# print('train saved.')
# validation_use.to_csv(DATASET_PATH + 'temp_validation_all_comp_encode_id.csv', index = False, compression='gzip')
# print('val saved.')
# test.to_csv(DATASET_PATH + 'temp_test_all_comp_encode_id.csv', index = False, compression='gzip')
# print('test saved.')


# In[ ]:


def new_artistname_related(train_data, val_data, test_data):
    gc.collect()
    temp_data = pd.concat([train_data, val_data, test_data], axis=0, join="outer")

    temp_data['avg_active_per_msno'] = temp_data[['active_1', 'active_2', 'active_3', 'active_4']].mean(axis = 1).astype(np.float32)
    
    grouped = temp_data[['msno', 'artist_name']].groupby(['artist_name'])
    
    num_people_per_artist = grouped['msno'].agg(lambda x: x.nunique())
    num_people_per_artist = num_people_per_artist.reset_index()
    num_people_per_artist.columns = ['artist_name', 'num_people_per_artist']
    #print(num_people_per_artist)
    sum_people_active_per_artist = temp_data[['artist_name', 'avg_active_per_msno']].groupby(['artist_name'])['avg_active_per_msno'].agg('mean')
    sum_people_active_per_artist = sum_people_active_per_artist.reset_index()
    sum_people_active_per_artist.columns = ['artist_name', 'sum_people_active_per_artist']
    #print(sum_people_active_per_artist)
    train_data = train_data.merge(right = num_people_per_artist, how = 'left', on='artist_name')
    val_data = val_data.merge(right = num_people_per_artist, how = 'left', on='artist_name')
    test_data = test_data.merge(right = num_people_per_artist, how = 'left', on='artist_name')
    
    train_data = train_data.merge(right = sum_people_active_per_artist, how = 'left', on='artist_name')
    val_data = val_data.merge(right = sum_people_active_per_artist, how = 'left', on='artist_name')
    test_data = test_data.merge(right = sum_people_active_per_artist, how = 'left', on='artist_name')
    
    return train_data[['num_people_per_artist', 'sum_people_active_per_artist']], val_data[['num_people_per_artist', 'sum_people_active_per_artist']], test_data[['num_people_per_artist', 'sum_people_active_per_artist']]
  


# In[ ]:


gc.collect()
train_use[['num_people_per_artist', 'sum_people_active_per_artist']], validation_use[['num_people_per_artist', 'sum_people_active_per_artist']], test[['num_people_per_artist', 'sum_people_active_per_artist']] = new_artistname_related(train_use[['artist_name', 'msno', 'active_1', 'active_2', 'active_3', 'active_4']],                                                                                          validation_use[['artist_name', 'msno', 'active_1', 'active_2', 'active_3', 'active_4']],                                                                                          test[['artist_name', 'msno', 'active_1', 'active_2', 'active_3', 'active_4']])

gc.collect()


# In[ ]:


time_wnd = [2018, 0, 2000, 2010, 2014, 2018]
def cal_artist_popular(train_data, val_data, test_data):
    all_data = pd.concat([train_data[['song_id', 'song_year', 'msno']], val_data[['song_id', 'song_year', 'msno']], test_data[['song_id', 'song_year', 'msno']]], axis=0, join="inner")
    #all_data['song_id'] = pd.to_numeric(all_data['song_id'], downcast='unsigned')
    #all_data['msno'] = pd.to_numeric(all_data['msno'], downcast='unsigned')
    for index, _ in enumerate(time_wnd[:-1]):
        begin_time, end_time = time_wnd[index] < time_wnd[index+1] and (time_wnd[index], time_wnd[index+1]) or (time_wnd[index+1], time_wnd[index])
#         begin_time = time_wnd[index]
#         end_time = time_wnd[index+1]
        select_data = all_data[all_data['song_year'].map(lambda x: x>=begin_time and x < end_time)]
        
        #select_data['target'] = pd.to_numeric(select_data['target'], downcast='signed')
        
        grouped = select_data[['song_id', 'msno']].groupby(['song_id'])

        count_song = grouped.agg(['count'])
        num_people_per_song = grouped.agg({"msno": lambda x: np.log(x.nunique()+1)})

        popularity = pd.concat([np.log(count_song+1), num_people_per_song], axis=1, join="inner")
        popularity.columns = ['popular_{}'.format(index), 'num_people_{}'.format(index)]
        popularity = popularity.reset_index(drop=False)
        train_data = train_data.merge(popularity, on='song_id', how ='left')
        test_data = test_data.merge(popularity, on='song_id', how ='left')
        val_data = val_data.merge(popularity, on='song_id', how ='left')
    return train_data, val_data, test_data


# In[ ]:


def new_msno_related(train_data, val_data, test_data):
    gc.collect()
    temp_data = pd.concat([train_data, val_data, test_data], axis=0, join="outer")

    temp_data['avg_active_per_msno'] = temp_data[['active_1', 'active_2', 'active_3', 'active_4']].mean(axis = 1).astype(np.float32)
    
    grouped = temp_data[['msno', 'artist_name']].groupby(['artist_name'])
    
    num_people_per_artist = grouped['msno'].agg(lambda x: x.nunique())
    num_people_per_artist = num_people_per_artist.reset_index()
    num_people_per_artist.columns = ['artist_name', 'num_people_per_artist']
    #print(num_people_per_artist)
    sum_people_active_per_artist = temp_data[['artist_name', 'avg_active_per_msno']].groupby(['artist_name'])['avg_active_per_msno'].agg('mean')
    sum_people_active_per_artist = sum_people_active_per_artist.reset_index()
    sum_people_active_per_artist.columns = ['artist_name', 'sum_people_active_per_artist']
    #print(sum_people_active_per_artist)
    train_data = train_data.merge(right = num_people_per_artist, how = 'left', on='artist_name')
    val_data = val_data.merge(right = num_people_per_artist, how = 'left', on='artist_name')
    test_data = test_data.merge(right = num_people_per_artist, how = 'left', on='artist_name')
    
    train_data = train_data.merge(right = sum_people_active_per_artist, how = 'left', on='artist_name')
    val_data = val_data.merge(right = sum_people_active_per_artist, how = 'left', on='artist_name')
    test_data = test_data.merge(right = sum_people_active_per_artist, how = 'left', on='artist_name')
    
    return train_data[['num_people_per_artist', 'sum_people_active_per_artist']], val_data[['num_people_per_artist', 'sum_people_active_per_artist']], test_data[['num_people_per_artist', 'sum_people_active_per_artist']]
  


# In[ ]:


for col in train_use.columns: print(col, ':', train_use[col].dtype, '; uinque values:', len(train_use[col].value_counts()))


# In[ ]:


for col in test.columns: print(col, ':', test[col].dtype, '; uinque values:', len(test[col].value_counts()))


# In[ ]:


for col in train_use.columns: print(col, ':', train_use[col].dtype, '; uinque values:', len(train_use[col].value_counts()))


# In[ ]:


gc.collect()
train_label = train_use['target']
#train_use.drop(['target'], axis=1).to_csv(DATASET_PATH + 'train.csv', index=False)
train_use = train_use.drop(['target'], axis=1)
#del train_use
gc.collect()


# In[ ]:


print(len(test_id), len(test), len(train_use), len(validation_use))
#del train_use_org, test_org, validation_use_org
gc.collect()


# In[ ]:


feature_list = list()
for col in test.columns:
    if col not in ['id']:
        feature_list.append(col)
print(feature_list)


# In[ ]:


catogory_list = list()
for col in test.columns:
    if col not in ['song_length', 'id']:
        if test[col].dtype.name == 'category':
            catogory_list.append(col)
print(catogory_list)


# In[ ]:


#用temp_train_all_comp_2 然后sum_merge_and_drop 和 drop


# In[ ]:


predictions = np.zeros(shape=[len(test)])

# features = ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city', 'bd', 'gender', 'registered_via', 'song_length', 'genre_ids', 'language', 'name', 'country', 'song_year', 'registration_year', 'registration_month', 'registration_day', 'expiration_year', 'expiration_month', 'expiration_day', 'days', 'composer_score', 'composer', 'lyricist_score', 'lyricist', 'artist_name_score', 'artist_name', 'popular_0', 'num_people_0', 'popular_1', 'num_people_1', 'popular_2', 'num_people_2', 'popular_3', 'num_people_3', 'popular_4', 'num_people_4', 'active_0', 'num_song_0', 'active_1', 'num_song_1', 'active_2', 'num_song_2', 'active_3', 'num_song_3', 'active_4', 'num_song_4', 'composer_by_city', 'composer_by_country', 'composer_by_language', 'lyricist_by_city', 'lyricist_by_country', 'lyricist_by_language', 'artist_name_by_city', 'artist_name_by_country', 'artist_name_by_language', 'song_id_by_city', 'msno_by_country', 'msno_by_language', 'genre_ids_score', 'genre_ids_popular', 'msno_is_in_topK_of_artist_name', 'msno_is_in_topK_of_genre_ids', 'artist_name_is_in_topK_of_genre_ids', 'artist_name_is_in_topK_of_msno', 'source_screen_name_count', 'source_screen_name_mean_popular_0', 'source_screen_name_mean_popular_1', 'source_screen_name_mean_active_0', 'source_screen_name_mean_active_1', 'source_type_count', 'source_type_mean_popular_0', 'source_type_mean_popular_1', 'source_type_mean_active_0', 'source_type_mean_active_1', 'city_by_song_id', 'city_by_msno', 'city_by_genre_ids', 'city_by_artist_name', 'language_by_song_id', 'language_by_msno', 'language_by_genre_ids', 'language_by_artist_name']

# train_use_array = train_use[features].values
# labels = train_use['target'].values.astype('int').flatten()
# #del train_use  # delete dataframe to release memory 
# gc.collect()
# train_data = lgb.Dataset(train_use_array, labels, feature_name = features, categorical_feature = ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'gender', 'name', 'country', 'composer', 'lyricist', 'artist_name'])

#train_data = lgb.Dataset(DATASET_PATH + 'train.csv', label=train_label)
train_data = lgb.Dataset(train_use, label=train_label)
gc.collect()
val_data = lgb.Dataset(validation_use.drop(['target'],axis=1),label=validation_use['target'])
del validation_use
gc.collect()

params = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'learning_rate': 0.1 ,
    'verbose': 0,
    'num_leaves': 128,#108
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 1,
    'feature_fraction': 0.9,
    'feature_fraction_seed': 1,
    'max_bin': 128,
    'max_depth': 12,
    #'num_rounds': 800,
    'metric' : 'auc',
    } 

bst = lgb.train(params, train_data, 800, valid_sets=[val_data])

#del train_data, val_data
gc.collect()

predictions+=bst.predict(test.drop(['id'],axis=1))
print('cur fold finished.')

submission = pd.DataFrame({'id': test_id, 'target': predictions})
submission.to_csv(SUBMISSION_FILENAME.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),index=False)


# In[ ]:


print('Plot feature importances...')
ax = lgb.plot_importance(bst, max_num_features=20)
plt.show()
# last 400 0.696958 800 0.703245


# In[ ]:


