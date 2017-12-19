from scipy.sparse import coo_matrix
import lightfm
import numpy as np
import pandas as pd
import time
import os
import logging


class LightFMProducer(object):
    __TRAIN_FILE_NAME = 'train.csv'
    __TEST_FILE_NAME = 'test.csv'
    __SONG_FILE_NAME = 'songs.csv'

    __NUM_THREADS = 5
    __NUM_COMPONENTS = 1
    __NUM_EPOCHS = 3
    __ITEM_ALPHA = 1e-6

    def __init__(self, root='./data', num_components=1):
        self._root = os.path.expanduser(root)

        self._train_df = None
        self._test_df = None
        self._comb_df = None

        self._genre_mapping = None

        self.__NUM_COMPONENTS = num_components
        self._model = lightfm.LightFM(loss='warp',
                                      item_alpha=LightFMProducer.__ITEM_ALPHA,
                                      no_components=LightFMProducer.__NUM_COMPONENTS)

    def prepare(self):
        start = time.time()
        # load train & test set
        self._train_df = pd.read_csv(os.path.join(self._root, self.__TRAIN_FILE_NAME))
        self._test_df = pd.read_csv(os.path.join(self._root, self.__TEST_FILE_NAME))
        songs_df = pd.read_csv(os.path.join(self._root, self.__SONG_FILE_NAME))

        self._train_df = self._train_df.merge(songs_df, on='song_id', how='left')
        self._test_df = self._test_df.merge(songs_df, on='song_id', how='left')

        self._train_df = self._train_df[['msno', 'song_id', 'genre_ids', 'target']]
        self._test_df = self._test_df[['msno', 'song_id', 'genre_ids']]
        self._comb_df = self._train_df.append(self._test_df)

        for column in self._train_df.columns:
            if self._train_df[column].dtype == object:
                self._train_df[column] = self._train_df[column].astype('category')

        for column in self._test_df.columns:
            if self._test_df[column].dtype == object:
                self._test_df[column] = self._test_df[column].astype('category')

        for column in self._comb_df.columns:
            if self._comb_df[column].dtype == object:
                self._comb_df[column] = self._comb_df[column].astype('category')

        self.__calculate_genre_mapping()
        self.__build_fm_model()

        logging.info("prepare in %0.2fs" % (time.time() - start))

    def compute(self, is_train):
        assert (isinstance(is_train, bool))

        if is_train:
            return self.generate_genre_latent_embeddings(self._train_df)
        else:
            return self.generate_genre_latent_embeddings(self._test_df)

    def generate_genre_latent_embeddings(self, df):
        item_csr_matrix = self.__get_csr_matrix(df)
        latent_factors = self._model.get_item_representations(item_csr_matrix)[1]

        item_factors_df = pd.DataFrame(data=latent_factors)
        item_factors_df.columns = ['item_fac_' + str(i) for i in range(self.__NUM_COMPONENTS)]

        df = df.join(item_factors_df)

        return df

    def __calculate_genre_mapping(self):
        genre_set = set()

        gid = 0
        for x in self._comb_df['genre_ids']:
            if x is not np.nan and x != 'nan':
                for item in str(x).strip().split('|'):
                    if str(item).isdigit():
                        genre_set.add(int(item))
            gid += 1

        arr = []
        for item in genre_set:
            arr += [item]
        arr.sort()

        mapping = {}
        gid = 0
        for item in arr:
            mapping[item] = gid
            gid += 1
        self._genre_mapping = mapping

    def __build_fm_model(self):
        train_csr_matrix = coo_matrix((self._train_df['target'].astype(float),
                                       (self._train_df['msno'].cat.codes,
                                        self._train_df['song_id'].cat.codes)))
        print(train_csr_matrix.todense().shape)

        item_csr_matrix = self.__get_csr_matrix(self._train_df)

        self._model.fit(train_csr_matrix,
                        item_features=item_csr_matrix,
                        epochs=LightFMProducer.__NUM_EPOCHS,
                        num_threads=LightFMProducer.__NUM_THREADS)

    def __get_csr_matrix(self, csr_df):
        assert self._genre_mapping is not None, 'build genre_mapping fist!'

        xs = []
        ys = []
        ranking = []
        gid = 0
        for x in csr_df['genre_ids']:
            if x is not np.nan and x != 'nan':
                for item in str(x).strip().split('|'):
                    if str(item).isdigit():
                        xs.append(gid)
                        ys.append(self._genre_mapping[int(item)])
                        ranking.append(1)
            gid += 1

        return coo_matrix((ranking, (xs, ys)))

    @property
    def train_df(self):
        return self._train_df

    @property
    def test_df(self):
        return self._test_df

