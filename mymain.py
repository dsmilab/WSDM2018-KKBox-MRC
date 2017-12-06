import seaborn as sns
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import datetime as dt
import random

import json
import os.path
import gc


def one_hot_encode_system_tab(x):
    return 1 if x == 'my library' else 0


def one_hot_encode_screen_name(x):
    return 1 if x == 'Local playlist more' or x == 'My library' else 0


def one_hot_encode_source_type(x):
    return 1 if x == 'local-library' or x == 'local-playlist' else 0


def one_hot_encode_source(x):
    return 1 if x >= 0.6 else 0

def parse_str_to_date(date_str):
    # [format] yyyymmdd
    date_str = str(date_str)
    assert(isinstance(date_str, str))
    assert(len(date_str) == 8)
    
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:])
    return dt.date(year, month, day)
    
           
def transform_two_dates_to_days(row):
    start = parse_str_to_date(row['registration_init_time'])
    end = parse_str_to_date(row['expiration_date'])
    delta = end - start
    return delta.days


def transform_bd_outliers(bd):
    # figure is from "exploration"
    if bd >= 120 or bd <= 7:
        return 'nan'
    mean = 28.99737187910644
    std = 9.538470787507382
    return bd if abs(bd - mean) <= 3 * std else 'nan'


def transform_outliers(x, mean, std):
    return x if np.abs(x - mean) <= 3 * std else -1


def one_hot_encode_via(x):
    return 0 if x == 4 else 1


def transform_init_time_to_ym(time):
    time_str = str(time)
    year = int(time_str[:4])
    month = int(time_str[4:6])
    return int("%04d%02d" % (year, month))

# def custom_gender_random_seed(x):
#    if x is not np.nan:
#        return x
#    return random.choice(['female', 'male'])
# reference http://isrc.ifpi.org/en/isrc-standard/code-syntax
def transform_isrc_to_year(isrc):
    if type(isrc) != str:
        return np.nan
    # this year 2017
    suffix = int(isrc[5:7])
    
    return 1900 + suffix if suffix > 17 else 2000 + suffix


def transform_isrc_to_country(isrc):
    if type(isrc) != str:
        return np.nan
    country = isrc[:2]
    
    return country


def transform_isrc_to_reg(isrc):
    if type(isrc) != str:
        return np.nan
    registration = isrc[2:5]
    
    return registration


def transfrom_isrc_to_desig(isrc):
    if type(isrc) != str:
        return np.nan
    designation = isrc[7:]
    
    return designation


def one_hot_encode_year(x):
    return 1 if 2013 <= float(x) <= 2017 else 0


def one_hot_encode_country(x):
    return 1 if x == 'TW' or x == 'CN' or x == 'HK' else 0

def parse_splitted_category_to_number(x):
    if x is np.nan:
        return 0
    
    x = str(x)
    x.replace('/', '|')
    x.replace(';', '|')
    x.replace('\\', '|')
    x.replace(' and ', '|')
    x.replace('&', '|')
    x.replace('+', '|')
    return x.count('|') + 1


def one_hot_encode_lang(x):
    return 1 if x in [-1, 17, 45] else 0


def my_output():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    songs_df = pd.read_csv('data/songs.csv')
    song_extra_info_df = pd.read_csv('data/song_extra_info.csv')
    members_df = pd.read_csv('data/members.csv')
    train_df['source_system_tab'].fillna('others', inplace=True)
    test_df['source_system_tab'].fillna('others', inplace=True)

    train_df['source_screen_name'].fillna('others', inplace=True)
    test_df['source_screen_name'].fillna('others', inplace=True)

    train_df['source_type'].fillna('nan', inplace=True)
    test_df['source_type'].fillna('nan', inplace=True)

    assert(~train_df.isnull().any().any())
    assert(~test_df.isnull().any().any())
    train_df['source_merged'] = train_df['source_system_tab'].map(str) + ' | ' + train_df['source_screen_name'].map(str) + ' | ' + train_df['source_type'].map(str) 
    test_df['source_merged'] = test_df['source_system_tab'].map(str) + ' | ' + test_df['source_screen_name'].map(str) + ' | ' + test_df['source_type'].map(str) 

    count_df = train_df[['source_merged', 'target']].groupby('source_merged').agg(['mean', 'count'])
    count_df.reset_index(inplace=True)
    count_df.columns = ['source_merged', 'source_replay_pb', 'source_replay_count']
    
    train_df = train_df.merge(count_df, on='source_merged', how='left')
    test_df = test_df.merge(count_df, on='source_merged', how='left')

    train_df['1h_source'] = train_df['source_replay_pb'].apply(one_hot_encode_source)
    test_df['1h_source'] = test_df['source_replay_pb'].apply(one_hot_encode_source)

    train_df.drop(['source_merged', 'source_replay_pb', 'source_replay_count'], axis=1, inplace=True)
    test_df.drop(['source_merged', 'source_replay_pb', 'source_replay_count'], axis=1, inplace=True)

    train_df['1h_system_tab'] = train_df['source_system_tab'].apply(one_hot_encode_system_tab)
    train_df['1h_screen_name'] = train_df['source_screen_name'].apply(one_hot_encode_screen_name)
    train_df['1h_source_type'] = train_df['source_type'].apply(one_hot_encode_source_type)

    test_df['1h_system_tab'] = test_df['source_system_tab'].apply(one_hot_encode_system_tab)
    test_df['1h_screen_name'] = test_df['source_screen_name'].apply(one_hot_encode_screen_name)
    test_df['1h_source_type'] = test_df['source_type'].apply(one_hot_encode_source_type)

    # never drop, important
    # train_df.drop(["source_system_tab", "source_screen_name", "source_type"], axis=1, inplace=True)
    # test_df.drop(["source_system_tab", "source_screen_name", "source_type"], axis=1, inplace=True)
    members_df['membership_days'] = members_df.apply(transform_two_dates_to_days, axis=1)

    members_df['registration_init_year'] = members_df['registration_init_time'].apply(lambda x: int(str(x)[:4]))
    members_df['registration_init_month'] = members_df['registration_init_time'].apply(lambda x: int(str(x)[4:6]))

    members_df['expiration_date_year'] = members_df['expiration_date'].apply(lambda x: int(str(x)[:4]))
    members_df['expiration_date_month'] = members_df['expiration_date'].apply(lambda x: int(str(x)[4:6]))

    members_df.drop(['registration_init_time'], axis=1, inplace=True)

    members_df['bd'] = members_df['bd'].apply(transform_bd_outliers)

    members_df['gender'].fillna('nan', inplace=True)

    members_df['1h_via'] = members_df['registered_via'].apply(one_hot_encode_via)

    assert(~members_df.isnull().any().any())
    members_df.head(15)
    
    song_extra_info_df['song_year'] = song_extra_info_df['isrc'].apply(transform_isrc_to_year)
    # song_extra_info_df['song_country'] = song_extra_info_df['isrc'].apply(transform_isrc_to_country)
    # song_extra_info_df['song_registration'] = song_extra_info_df['isrc'].apply(transform_isrc_to_reg)
    # song_extra_info_df['song_designation'] = song_extra_info_df['isrc'].apply(transfrom_isrc_to_desig)

    song_extra_info_df['1h_song_year'] = song_extra_info_df['song_year'].apply(one_hot_encode_year)
    # song_extra_info_df['1h_song_country'] = song_extra_info_df['song_country'].apply(one_hot_encode_country)

    song_extra_info_df.drop(['isrc', 'name'], axis=1, inplace=True)

    song_extra_info_df['song_year'].fillna(2017, inplace=True)
    # song_extra_info_df['song_registration'].fillna('***', inplace=True)

    assert(~song_extra_info_df.isnull().any().any())
    song_extra_info_df.head(15)
    
    songs_df['genre_count'] = songs_df['genre_ids'].apply(parse_splitted_category_to_number)
    songs_df['composer_count'] = songs_df['composer'].apply(parse_splitted_category_to_number)
    songs_df['lyricist_count'] = songs_df['lyricist'].apply(parse_splitted_category_to_number)

    songs_df['1h_lang'] = songs_df['language'].apply(one_hot_encode_lang)

    songs_df['1h_song_length'] = songs_df['song_length'].apply(lambda x: 1 if x <= 239738 else 0)

    songs_df['language'].fillna('nan', inplace=True)
    songs_df['composer'].fillna('nan', inplace=True)
    songs_df['lyricist'].fillna('nan', inplace=True)
    songs_df['genre_ids'].fillna('nan', inplace=True)
    # songs_df.drop(['language'], axis=1, inplace=True)
    assert(~songs_df.isnull().any().any())
    songs_df.head(15)
    
    songs_df = songs_df.merge(song_extra_info_df, on='song_id', how='left')

    train_df = train_df.merge(songs_df, on='song_id', how='left')
    test_df = test_df.merge(songs_df, on='song_id', how='left')

    train_df = train_df.merge(members_df, on='msno', how='left')
    test_df = test_df.merge(members_df, on='msno', how='left')

    comb_df = train_df.append(test_df)

    train_df.info()
    test_df.info()
    
    count_df = train_df['song_id'].value_counts().reset_index()
    count_df.columns = ['song_id', 'play_count']

    train_df = train_df.merge(count_df, on='song_id', how='left')
    train_df['play_count'].fillna(0, inplace=True)

    count_df = comb_df['song_id'].value_counts().reset_index()
    count_df.columns = ['song_id', 'play_count']

    test_df = test_df.merge(count_df, on='song_id', how='left')
    test_df['play_count'].fillna(0, inplace=True)
    
    track_count_df = train_df[['song_id',
                           'artist_name']].drop_duplicates('song_id')
    track_count_df = track_count_df.groupby('artist_name').agg(
        'count').reset_index()
    track_count_df.columns = ['artist_name', 'track_count']
    track_count_df = track_count_df.sort_values('track_count', ascending=False)
    track_count_df.head(10)
    
    artist_count_df = train_df[['artist_name',
                            'target']].groupby('artist_name').agg(
                                ['mean', 'count']).reset_index()
    artist_count_df.columns = ['artist_name', 'replay_pb', 'play_count']

    artist_count_df = artist_count_df.merge(
        track_count_df, on='artist_name', how='left')
    artist_count_df['track_count'].fillna(0, inplace=True)

    train_df = train_df.merge(
        artist_count_df[['artist_name', 'track_count']],
        on='artist_name',
        how='left')

    ####

    artist_count_df = comb_df[['artist_name',
                               'target']].groupby('artist_name').agg(
                                   ['mean', 'count']).reset_index()
    artist_count_df.columns = ['artist_name', 'replay_pb', 'play_count']

    artist_count_df = artist_count_df.merge(
        track_count_df, on='artist_name', how='left')
    artist_count_df['track_count'].fillna(0, inplace=True)

    test_df = test_df.merge(
        artist_count_df[['artist_name', 'track_count']],
        on='artist_name',
        how='left')
    
    for column in train_df.columns:
        if train_df[column].dtype == object:
            train_df[column] = train_df[column].astype('category')
    for column in test_df.columns:
        if test_df[column].dtype == object:
            test_df[column] = test_df[column].astype('category')
    
    return train_df, test_df
