# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
import time
import os
import logging
from abc import abstractmethod

from implicit.als import AlternatingLeastSquares
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix, linalg
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity

LOG_FORMAT = '%(asctime)s %(levelname)s << %(message)s'


class DataProceesor(object):

    def __init__(self):
        return

    @abstractmethod
    def parse(self, df):
        raise NotImplementedError("Please implement method \'parse()\'.")

    @staticmethod
    def process(df, command):
        start = time.time()

        res = None
        if command == 'members':
            res = MembersProcessor().parse(df)
        elif command == 'songs':
            res = SongsProcessor().parse(df)
        logging.info("parse %s_df in %0.2fs" % (command, time.time() - start))

        return res


class SongsProcessor(DataProceesor):

    def __init__(self):
        super(SongsProcessor, self).__init__()
        return

    def parse(self, df):
        df['artist_name'].fillna('no_artist', inplace=True)
        df['is_featured'] = df['artist_name'].apply(SongsProcessor.__is_featured).astype(np.int8)

        # >> duplicate
        df['artist_count'] = df['artist_name'].apply(SongsProcessor.__artist_count).astype(np.int8)

        df['artist_composer'] = (df['artist_name'] == df['composer'])
        df['artist_composer'] = df['artist_composer'].astype(np.int8)

        # if artist, lyricist and composer are all three same
        df['artist_composer_lyricist'] = ((df['artist_name'] == df['composer']) &
                                          (df['artist_name'] == df['lyricist']) &
                                          (df['composer'] == df['lyricist']))
        df['artist_composer_lyricist'] = df['artist_composer_lyricist'].astype(np.int8)

        # >> duplicate
        df['song_lang_boolean'] = df['language'].apply(SongsProcessor.__song_lang_boolean).astype(np.int8)

        # howeverforever
        df['genre_count'] = df['genre_ids'].apply(SongsProcessor.__parse_splitted_category_to_number)
        df['composer_count'] = df['composer'].apply(SongsProcessor.__parse_splitted_category_to_number)
        df['lyricist_count'] = df['lyricist'].apply(SongsProcessor.__parse_splitted_category_to_number)

        df['1h_lang'] = df['language'].apply(SongsProcessor.__one_hot_encode_lang)

        df['1h_song_length'] = df['song_length'].apply(lambda x: 1 if x <= 239738 else 0)

        df['language'].fillna('nan', inplace=True)
        df['composer'].fillna('nan', inplace=True)
        df['lyricist'].fillna('nan', inplace=True)
        df['genre_ids'].fillna('nan', inplace=True)
        # df.drop(['language'], axis=1, inplace=True)
        assert(~df.isnull().any().any())

        return df

    @staticmethod
    def __is_featured(x):
        return 1 if 'feat' in str(x) else 0

    @staticmethod
    def __artist_count(x):
        return 0 if x == 'no_artist' else x.count('and') + x.count(',') + x.count('feat') + x.count('&')

    @staticmethod
    def __song_lang_boolean(x):
        # is song language 17 or 45.
        return 1 if '17.0' in str(x) or '45.0' in str(x) else 0

    @staticmethod
    def __parse_splitted_category_to_number(x):
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

    @staticmethod
    def __one_hot_encode_lang(x):
        return 1 if x in [-1, 17, 45] else 0


class MembersProcessor(DataProceesor):

    def __init__(self):
        super(MembersProcessor, self).__init__()
        return

    def parse(self, df):
        df['membership_days'] = df['expiration_date'].subtract(df['registration_init_time']).dt.days.astype(int)

        df['registration_year'] = df['registration_init_time'].dt.year
        df['registration_month'] = df['registration_init_time'].dt.month
        df['registration_date'] = df['registration_init_time'].dt.day

        df['expiration_year'] = df['expiration_date'].dt.year
        df['expiration_month'] = df['expiration_date'].dt.month
        df['expiration_date'] = df['expiration_date'].dt.day
        df = df.drop(['registration_init_time'], axis=1)

        # howeverforever
        df['bd'] = df['bd'].apply(MembersProcessor.__transform_bd_outliers)
        df['gender'].fillna('nan', inplace=True)
        df['1h_via'] = df['registered_via'].apply(MembersProcessor.__one_hot_encode_via)
        assert (~df.isnull().any().any())

        return df

    @staticmethod
    def __transform_bd_outliers(bd):
        # figure is from "exploration"
        if bd >= 120 or bd <= 7:
            return 'nan'
        mean = 28.99737187910644
        std = 9.538470787507382
        return bd if abs(bd - mean) <= 3 * std else 'nan'

    @staticmethod
    def __one_hot_encode_via(x):
        return 0 if x == 4 else 1


class FeatureProcessor(object):

    __SONGS_FILE_NAME = 'songs.csv'
    __SONG_EXTRA_FILE_NAME = 'song_extra_info.csv'
    __MEMBERS_FILE_NAME = 'members.csv'
    __TRAIN_FILE_NAME = 'train.csv'
    __TEST_FILE_NAME = 'test.csv'

    def __init__(self, root='./data'):
        assert os.path.exists(root), '%s not exists!' % root
        self.root = os.path.expanduser(root)

        self.songs_df = None
        self.song_extra_df = None
        self.members_df = None
        self.train_df = None
        self.test_df = None

    def load_raw(self):
        start = time.time()

        # load train & test set
        self.train_df = pd.read_csv(os.path.join(self.root, self.__TRAIN_FILE_NAME))
        self.test_df = pd.read_csv(os.path.join(self.root, self.__TEST_FILE_NAME))

        # load song & member set
        self.songs_df = pd.read_csv(os.path.join(self.root, self.__SONGS_FILE_NAME))
        self.song_extra_df = pd.read_csv(os.path.join(self.root, self.__SONG_EXTRA_FILE_NAME))
        self.members_df = pd.read_csv(os.path.join(self.root, self.__MEMBERS_FILE_NAME),
                                      parse_dates=['registration_init_time','expiration_date'])

        logging.info("load raw data in %0.2fs" % (time.time() - start))

    def pre_process(self):
        self.members_df = DataProceesor.process(self.members_df, 'members')
        self.songs_df = DataProceesor.process(self.songs_df, "songs")


def main():
    fp = FeatureProcessor(root='../data')
    fp.load_raw()
    fp.pre_process()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt='%H:%M:%S')
    main()
