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
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns].to_dict(orient='record')

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

class FeatureProcesser(object):
    songs_file = 'songs.csv'
    extra_file = 'song_extra_info.csv'
    members_file = 'members.csv'
    train_file = 'train.csv'
    test_file = 'test.csv'

    def __init__(self, root='./data'):

        assert os.path.exists(root), '%s not exists!' % root
        self.root = os.path.expanduser(root)
        train, test = self._load_raw()
        self._process_member()
        self._process_extra()

        self._compute_msno_song_similarity(train, test)

        train = self._preprocess(train)
        test = self._preprocess(test)

        self._mean_song_length = np.mean(train['song_length'])

        # number of times a song has been played before
        self._dict_count_song_played_train = {k: v for k, v in train['song_id'].value_counts().items()}
        self._dict_count_song_played_test = {k: v for k, v in test['song_id'].value_counts().items()}
        # number of times the artist has been played
        self._dict_count_artist_played_train = {k: v for k, v in train['artist_name'].value_counts().items()}
        self._dict_count_artist_played_test = {k: v for k, v in test['artist_name'].value_counts().items()}

        train['count_artist_played'] = train['artist_name'].apply(self._count_artist_played).astype(np.int64)
        test['count_artist_played'] = test['artist_name'].apply(self._count_artist_played).astype(np.int64)

        train['count_song_played'] = train['song_id'].apply(self._count_song_played).astype(np.int64)
        test['count_song_played'] = test['song_id'].apply(self._count_song_played).astype(np.int64)

        train = self._add_new_feature(train)
        test = self._add_new_feature(test)

        # total_genre_ids = pd.concat([train.genre_ids, test.genre_ids])
        # self.genres = np.unique('|'.join(total_genre_ids).split('|'))

        # train = self._add_genre_feature(train)
        # test = self._add_genre_feature(test)

        train.fillna('na_for_later', inplace=True)
        test.fillna('na_for_later', inplace=True)

        for col in train.columns:
            if train[col].dtype == object:
                train[col] = train[col].astype('category')
                test[col] = test[col].astype('category')

        self.train = train
        self.test = test

    def load(self):
        return self.train, self.test

    def _transform_isrc_to_year(self, isrc):
        if type(isrc) != str:
            return np.nan
        # this year 2017
        suffix = int(isrc[5:7])
        return 1900 + suffix if suffix > 17 else 2000 + suffix

    def _load_raw(self):
        start = time.time()

        self.songs = pd.read_csv(os.path.join(self.root, self.songs_file))
        self.extra = pd.read_csv(os.path.join(self.root, self.extra_file))
        self.members = pd.read_csv(os.path.join(self.root, self.members_file),
                                   parse_dates=['registration_init_time','expiration_date'])

        train_raw = pd.read_csv(
                os.path.join(self.root, self.train_file))

        test_raw = pd.read_csv(
                os.path.join(self.root, self.test_file))

        logging.debug("load raw data in %0.2fs" % (time.time() - start))

        return train_raw, test_raw

    def _preprocess(self, df):

        start = time.time()
        df = df.merge(self.songs, on='song_id', how='left')
        df = df.merge(self.members, on='msno', how='left')
        df = df.merge(self.extra, on = 'song_id', how = 'left')

        df.song_length.fillna(200000,inplace=True)
        df.song_length = df.song_length.astype(np.uint32)
        # df.song_id = df.song_id.astype('category')
        logging.debug("preprocess in %0.2fs" % (time.time() - start))

        return df

    def _genre_id_count(self, x):
        if x == 'no_genre_id':
            return 0
        else:
            return x.count('|') + 1

    def _lyricist_count(self, x):
        if x == 'no_lyricist':
            return 0
        else:
            return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
        return sum(map(x.count, ['|', '/', '\\', ';']))

    def _composer_count(self, x):
        if x == 'no_composer':
            return 0
        else:
            return sum(map(x.count, ['|', '/', '\\', ';'])) + 1

    def _is_featured(self, x):
        if 'feat' in str(x) :
            return 1
        return 0

    def _artist_count(self, x):
        if x == 'no_artist':
            return 0
        else:
            return x.count('and') + x.count(',') + x.count('feat') + x.count('&')

    def _song_lang_boolean(self, x):
        # is song language 17 or 45.
        if '17.0' in str(x) or '45.0' in str(x):
            return 1
        return 0

    def _is_2017(self, x):
        if x == 2017.0:
            return 1
        return 0

    def _smaller_song(self, x):
        if x < self._mean_song_length:
            return 1
        return 0

    def _count_song_played(self, x):
        try:
            return self._dict_count_song_played_train[x]
        except KeyError:
            try:
                return self._dict_count_song_played_test[x]
            except KeyError:
                return 0

    def _count_artist_played(self, x):
        try:
            return self._dict_count_artist_played_train[x]
        except KeyError:
            try:
                return self._dict_count_artist_played_test[x]
            except KeyError:
                return 0

    def _find_genre(self, g_list, g):
        return True if g in g_list else False

    def _add_genre_feature(self, df):

        start = time.time()
        genre_ids = df.genre_ids.apply(lambda x: x.split('|'))
        for g in self.genres:
            df['g_' + g] = genre_ids.apply(lambda x: self._find_genre(x, g))

        logging.debug("add genre features in %0.2fs" % (time.time() - start))

        return df

    def _add_new_feature(self, df):

        start = time.time()
        df['genre_ids'].fillna('no_genre_id',inplace=True)
        df['genre_ids_count'] = df['genre_ids'].apply(self._genre_id_count).astype(np.int8)

        df['lyricist'].fillna('no_lyricist',inplace=True)
        df['lyricists_count'] = df['lyricist'].apply(self._lyricist_count).astype(np.int8)

        df['composer'].fillna('no_composer',inplace=True)
        df['composer_count'] = df['composer'].apply(self._composer_count).astype(np.int8)

        df['artist_name'].fillna('no_artist',inplace=True)
        df['is_featured'] = df['artist_name'].apply(self._is_featured).astype(np.int8)

        df['artist_count'] = df['artist_name'].apply(self._artist_count).astype(np.int8)

        # if artist is same as composer
        df['artist_composer'] = (df['artist_name'] == df['composer']).astype(np.int8)

        # if artist, lyricist and composer are all three same
        df['artist_composer_lyricist'] = ((df['artist_name'] == df['composer']) & (df['artist_name'] == df['lyricist']) & (df['composer'] == df['lyricist'])).astype(np.int8)

        df['song_lang_boolean'] = df['language'].apply(self._song_lang_boolean).astype(np.int8)
        df['smaller_song'] = df['song_length'].apply(self._smaller_song).astype(np.int8)

        df['is_2017'] = df['song_year'].apply(self._is_2017).astype(np.int8)

        logging.debug("add new features in %0.2fs" % (time.time() - start))

        return df

    def _process_member(self):

        self.members['membership_days'] = self.members['expiration_date'].subtract(self.members['registration_init_time']).dt.days.astype(int)

        self.members['registration_year'] = self.members['registration_init_time'].dt.year
        self.members['registration_month'] = self.members['registration_init_time'].dt.month
        self.members['registration_date'] = self.members['registration_init_time'].dt.day

        self.members['expiration_year'] = self.members['expiration_date'].dt.year
        self.members['expiration_month'] = self.members['expiration_date'].dt.month
        self.members['expiration_date'] = self.members['expiration_date'].dt.day
        self.members = self.members.drop(['registration_init_time'], axis=1)

    def _process_extra(self):
        self.extra['song_year'] = self.extra['isrc'].apply(self._transform_isrc_to_year)
        self.extra.drop(['name', 'isrc'], axis=1, inplace=True)

    def _compute_msno_song_similarity(self, train, test):

        start = time.time()
        member_feature = ['city',
                        'bd',
                        'gender',
                        'registered_via',
                        'expiration_date',
                        'membership_days',
                        'registration_year',
                        'registration_month',
                        'registration_date',
                        'expiration_year',
                        'expiration_month']

        song_feature = ['genre_ids',
                     'artist_name',
                     'language',
                     'composer',
                     'lyricist',
                     'song_year']

        member_pipeline = Pipeline([
                ('extract', ColumnSelector(member_feature)),
                ('dicVect', DictVectorizer())])
        song_pipeline = Pipeline([
                ('extract', ColumnSelector(song_feature)),
                ('dicVect', DictVectorizer())])

        songs = self.songs.merge(self.extra, on = 'song_id', how = 'left').fillna('test')
        members = self.members.fillna('test')
        msno_ix = {v: i for i, v in enumerate(members.msno)}
        song_ix = {v: i for i, v in enumerate(songs.song_id)}

        msno_m = member_pipeline.fit_transform(members)
        logging.debug("transform members in %0.2fs" % (time.time() - start))

        start = time.time()
        song_m = song_pipeline.fit_transform(songs)
        logging.debug("transform songs in %0.2fs" % (time.time() - start))

        start = time.time()
        known_msno = list(train.msno.unique())
        known_song = list(train.song_id.unique())

        unknown_msno = list(set(test.msno.unique()) - set(known_msno))
        total_msno = float(len(unknown_msno))
        unknown_song = list(set(test.song_id.unique()) - set(known_song))
        total_song = float(len(unknown_song))

        unknown_msno_map, unknown_song_map = {}, {}
        print(total_msno)
        n = 0
        for i in unknown_msno:
            df = self._get_rank(msno_m, msno_ix[i], members.msno)
            for index, row in df.iterrows():
                if row['id'] in known_msno:
                    unknown_msno_map[i] = row['id']
                    break
            n += 1
            if (n + 1) % 100000: print('msno: %f %%\n' % ((n/total_msno) * 100))

        print(total_song)
        n = 0
        for i in unknown_song:
            df = self._get_rank(song_m, song_ix[i], songs.song_id)
            for index, row in df.iterrows():
                if row['id'] in known_song:
                    unknown_song_map[i] = row['id']
                    break
            n += 1
            if (n + 1) % 1000: print('song: %f %%\n' % ((n/total_song) * 100))

        logging.debug("transform all unknown data in %0.2fs" % (time.time() - start))
        return unknown_msno_map, unknown_song_map

    def _get_rank(self, model, w, id_list):
        result = cosine_similarity(model, model[w].toarray().reshape(1, -1)).reshape(1, -1)[0]
        return pd.DataFrame({'id': id_list, 'similarity': result}).sort_values(by='similarity', ascending=False)

class ImplicitProcesser(object):

    def __init__(self, feature_size=100, calculate_training_loss=False, save_dir='./model',
                 iterations=15, n_clusters=30, random_state=50, cluster=True):

        assert os.path.exists(save_dir), '%s not exists!' % save_dir
        self.save_dir = os.path.expanduser(save_dir)

        assert feature_size % 2 == 0, 'feature_size need to be an even number!'
        self.factors = int(feature_size / 2)

        self.calculate_training_loss = calculate_training_loss
        self.iterations = iterations
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster = cluster

    def fit(self, train_df, test_df, unknown_msno_map, unknown_song_map):

        self.train_raw = train_df[['msno', 'song_id', 'target']]
        self.test_raw = test_df[['msno', 'song_id']]

        self.unknown_msno_map = unknown_msno_map
        self.unknown_song_map = unknown_song_map

        self._process_train()
        self._process_test()

        self._fit_model(self.calculate_training_loss, self.iterations)

        train_factors = self._get_factors(train=True)
        test_factors = self._get_factors(train=False)

        if self.cluster:

            self._get_clusting_feature(self.n_clusters, self.random_state)
            train_group = self._get_group(train=True)
            test_group = self._get_group(train=False)

            train_add_feature = pd.concat([train_group, train_factors], axis=1, ignore_index=True)
            test_add_feature = pd.concat([test_group, test_factors], axis=1, ignore_index=True)
        else:
            train_add_feature = train_factors
            test_add_feature = test_factors

        y_train = train_df['target'].values
        train_df = train_df.drop(['target'], axis=1)
        # train_df = train_df.drop(['msno', 'song_id', 'target'], axis=1)

        ids = test_df['id'].values
        test_df = test_df.drop(['id'], axis=1)
        # test_df = test_df.drop(['msno', 'song_id', 'id'], axis=1)

        X_train = pd.concat([train_df, train_add_feature], axis=1, ignore_index=True)
        X_test = pd.concat([test_df, test_add_feature], axis=1, ignore_index=True)

        return X_train, y_train, X_test, ids

    def _get_clusting_feature(self, n_clusters, random_state):

        start = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

        self.item_group = kmeans.fit_predict(self.item_factors)
        self.user_group = kmeans.fit_predict(self.user_factors)
        logging.debug("add clusting feature in %0.2fs" % (time.time() - start))

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
            lambda x: self.msno_ix[x] if x in self.msno_ix.keys() else self.msno_ix[self.unknown_msno_map[x]]).astype("category")

        self.test_raw['song_ix'] = self.test_raw.song_id.apply(
            lambda x: self.song_ix[x] if x in self.song_ix.keys() else self.song_ix[self.unknown_song_map[x]]).astype("category")
        logging.debug("preprocess test data in %0.2fs" % (time.time() - start))

    def _fit_model(self, calculate_training_loss, iterations):

        start = time.time()
        model = AlternatingLeastSquares(factors=self.factors,
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

    def _get_factors(self, train=True):

        if train:
            df = self.train_raw
            song_factor = np.array([self.item_factors[i] for i in df.song_ix])
            msno_factor = np.array([self.user_factors[i] for i in df.msno_ix])

        else:
            df = self.test_raw
            song_factor = np.array([self.item_factors[i] if i != 'new' else np.full(self.factors, np.nan) for i in df.song_ix])
            msno_factor = np.array([self.user_factors[i] if i != 'new' else np.full(self.factors, np.nan) for i in df.msno_ix])

        return pd.concat([pd.DataFrame(song_factor), pd.DataFrame(msno_factor)], axis=1, ignore_index=True)

    def _get_group(self, train=True):

        df = pd.DataFrame()
        if train:
            df['song_group'] = self.train_raw.song_ix.apply(lambda x: self.item_group[x])
            df['msno_group'] = self.train_raw.msno_ix.apply(lambda x: self.user_group[x])
        else:
            df['song_group'] = self.test_raw.song_ix.apply(lambda x: self.item_group[x] if x != 'new' else np.nan)
            df['msno_group'] = self.test_raw.msno_ix.apply(lambda x: self.user_group[x] if x != 'new' else np.nan)

        return df

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

    def get_song_list(self):
        return self.song_list
