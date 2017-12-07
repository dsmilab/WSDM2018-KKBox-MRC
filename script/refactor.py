# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
import time
import os
import logging

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

        self.songs_df = pd.read_csv(os.path.join(self.root, self.__SONGS_FILE_NAME))
        self.song_extra_df = pd.read_csv(os.path.join(self.root, self.__SONG_EXTRA_FILE_NAME))
        self.members_df = pd.read_csv(os.path.join(self.root, self.__MEMBERS_FILE_NAME))

        self.train_df = pd.read_csv(os.path.join(self.root, self.__TRAIN_FILE_NAME))
        self.test_df = pd.read_csv(os.path.join(self.root, self.__TEST_FILE_NAME))

        logging.info("load raw data in %0.2fs" % (time.time() - start))


def main():
    fp = FeatureProcessor(root='../data')
    fp.load_raw()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt='%H:%M:%S')
    main()
