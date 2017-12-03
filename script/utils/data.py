# -*- coding: utf-8 -*-
from implicit.als import AlternatingLeastSquares
import numpy as np
import pandas as pd
import pickle
import time
import os
import logging
from scipy.sparse import coo_matrix, linalg
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset

class KKboxRSDataset(Dataset):

    def __init__(self, train=True, processor=None):
        self.train = train

        if self.train:
            self.train_data, self.train_labels = processor.load(train=self.train)
        else:
            self.test_data, self.test_labels = processor.load(train=self.train)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (vectors, target) where target is index of the target class.
        """
        if self.train:
            vectors, target = self.train_data[index], self.train_labels[index]
        else:
            vectors, target = self.test_data[index], self.test_labels[index]

        return vectors, target

class ImplicitProcesser(object):

    songs_file = 'songs.csv'
    extra_file = 'song_extra_info.csv'
    members_file = 'members.csv'
    train_file = 'train.csv'
    test_file = 'test.csv'
    # train_cols = ['msno', 'song_id', 'target']
    # test_cols = ['msno', 'song_id', 'id']
    rare_threshold = 3

    def __init__(self, root='./data', feature_size=100, real_test=False,
                 calculate_training_loss=False, save_dir='./model',
                 iterations=15, rm_rare=False):

        assert os.path.exists(root), '%s not exists!' % root
        self.root = os.path.expanduser(root)

        assert os.path.exists(save_dir), '%s not exists!' % save_dir
        self.save_dir = os.path.expanduser(save_dir)

        assert feature_size % 2 == 0, 'feature_size need to be an even number!'
        factors = int(feature_size / 2)

        self.real_test = real_test
        self.rm_rare = rm_rare
        self._load_raw()
        self._process_train()
        self._process_test()
        self._get_latent_factor(factors,
                                calculate_training_loss,
                                iterations)

    def _transform_isrc_to_year(self, isrc):
        if type(isrc) != str:
            return np.nan
        # this year 2017
        suffix = int(isrc[5:7])
        return 1900 + suffix if suffix > 17 else 2000 + suffix

    def _split_unknown(self, test_raw):

        m = set(test_raw.msno) - set(self.train_raw.msno)
        s = set(test_raw.song_id) - set(self.train_raw.song_id)

        is_known_m = lambda x: x in m
        is_known_s = lambda x: x in s

        unknown = test_raw[test_raw.msno.apply(is_known_m) | test_raw.song_id.apply(is_known_s)]
        known_ix = test_raw.index.difference(unknown.index)

        return test_raw.loc[known_ix], unknown

    def _rm_train_rare(self, train_raw):

        song_count = train_raw.song_id.value_counts()
        r = song_count[song_count < self.rare_threshold].index
        is_not_rare = lambda x: x not in r
        train_raw = train_raw[train_raw.song_id.apply(is_not_rare)]

        return train_raw

    def _load_raw(self):

        start = time.time()
        songs = pd.read_csv(os.path.join(self.root, self.songs_file))
        extra = pd.read_csv(os.path.join(self.root, self.extra_file))

        extra['song_year'] = extra['isrc'].apply(self._transform_isrc_to_year)
        extra.drop(['name', 'isrc'], axis=1, inplace=True)
        self.songs = songs.merge(extra, on='song_id', how='left')
        # self.songs.index = self.songs.song_id

        self.members = pd.read_csv(os.path.join(self.root, self.members_file))

        train_raw = pd.read_csv(
                os.path.join(self.root, self.train_file))

        if self.real_test:
            test_raw = pd.read_csv(
                os.path.join(self.root, self.test_file))
        else:
            train_raw, test_raw = train_test_split(train_raw, random_state=50, shuffle=False)

        if self.rm_rare:
            self.train_raw = self._rm_train_rare(train_raw)
            logging.debug("%0.2f%% rare." % ((1 - (self.train_raw.shape[0] / train_raw.shape[0])) * 100))
        else:
            self.train_raw = train_raw

        # self.test_raw, self.unknown = self._split_unknown(test_raw)
        # logging.debug("%0.2f%% unknown." % ((self.unknown.shape[0] / test_raw.shape[0]) * 100))
        self.test_raw = test_raw

        logging.debug("load raw data in %0.2fs" % (time.time() - start))

    def _process_train(self):

        start = time.time()

        self.msno_list = list(self.train_raw.msno.unique())
        self.song_list = list(self.train_raw.song_id.unique())

        self.msno_ix = {v: i for i, v in enumerate(self.msno_list)}
        self.song_ix = {v: i for i, v in enumerate(self.song_list)}

        self.train_raw['msno_ix'] = self.train_raw.msno.apply(
            lambda x: self.msno_ix[x]).astype("category")

        self.train_raw['song_ix'] = self.train_raw.song_id.apply(
            lambda x: self.song_ix[x]).astype("category")

        # self.train_raw = self.train_raw[self.train_raw.target == 1]
        self.targets = coo_matrix((self.train_raw['target'].astype(float),
                                  (self.train_raw['song_ix'].cat.codes,
                                   self.train_raw['msno_ix'].cat.codes)))

        pickle.dump(self.msno_list, open(os.path.join(self.save_dir, 'msno_list.pkl'), 'wb'))
        pickle.dump(self.song_list, open(os.path.join(self.save_dir, 'song_list.pkl'), 'wb'))

        logging.debug("preprocess train data in %0.2fs" % (time.time() - start))

    def _process_test(self):

        start = time.time()

        self.test_raw['msno_ix'] = self.test_raw.msno.apply(
            lambda x: self.msno_ix[x] if x in self.msno_ix.keys() else 'new').astype("category")

        self.test_raw['song_ix'] = self.test_raw.song_id.apply(
            lambda x: self.song_ix[x] if x in self.song_ix.keys() else 'new').astype("category")
        logging.debug("preprocess test data in %0.2fs" % (time.time() - start))

    def _get_latent_factor(self, factors, calculate_training_loss, iterations):

        start = time.time()
        model = AlternatingLeastSquares(factors=factors,
                                        calculate_training_loss=calculate_training_loss,
                                        iterations=iterations)
        model.fit(self.targets)

        self.item_factors = normalize(model.item_factors)
        self.user_factors = normalize(model.user_factors)

        pickle.dump(model, open(os.path.join(self.save_dir, 'implicit_model.pkl'), 'wb'))

        logging.debug("train implicit model in %0.2fs" % (time.time() - start))

    def _get_top(self, model, w, top_n):
        result = cosine_similarity(model, model[w].reshape(1, -1)).reshape(1, -1)[0]
        return [(i, result[i]) for i in result.argsort()[::-1][:top_n + 1]]

    def _transform(self, train=True):

        if train:
            df = self.train_raw
            song_factor = np.array([self.item_factors[i] for i in df.song_ix])
            msno_factor = np.array([self.user_factors[i] for i in df.msno_ix])
        else:
            df = self.test_raw
            song_factor = np.array([self.item_factors[i] if i != 'new' else np.full(25, np.nan) for i in df.song_ix])
            msno_factor = np.array([self.user_factors[i] if i != 'new' else np.full(25, np.nan) for i in df.msno_ix])

        logging.debug('merge needed information')

        df = df.merge(self.songs, on='song_id', how='left')
        df = df.merge(self.members, on='msno', how='left')

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype('category')

        df = pd.concat([df, pd.DataFrame(song_factor)], axis=1)
        df = pd.concat([df, pd.DataFrame(msno_factor)], axis=1)

        # u = np.array(list(map(msnoix_to_factors, df.msno_ix)))
        # s = np.array(list(map(songix_to_factors, df.song_ix)))

        # features = np.column_stack((u, s))

        if not train and self.real_test:
            X = df.drop(['id'], axis=1)
            y = df['id'].values
        else:
            X = df.drop(['target'], axis=1)
            y = df['target'].values

        return X, y

    def print_similar_user(self, msno, top_n):

        print('-----')
        m = self.msno_ix[msno]

        for i, v in self._get_top(self.user_factors, m, top_n):
            print("%0.4f" % v, self.msno_list[i])

        print('-----')

    def print_similar_song(self, song_id, top_n):

        print('-----')
        s = self.song_ix[song_id]

        for i, v in self._get_top(self.item_factors, s, top_n):
            t = self.song_list[i]
            print('%0.4f' % v, self.songs.loc[t]['artist_name'], '-', self.extra.loc[t]['name'])

        print('-----')

    def get_unknown(self):
        return self.unknown

    def get_song_list(self):
        return self.song_list

    def load(self, train=True):
        return self._transform(train)
