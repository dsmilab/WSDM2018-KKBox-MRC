import numpy as np
import pandas as pd
import pickle
import implicit
import time
import os
import logging

# self-defined
from lib.FeatureProducer import FeatureProducer


LOG_FORMAT = '%(asctime)s %(levelname)s << %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt='%H:%M:%S')


class ImplicitProducer(object):

    __TRAIN_FILE_NAME = 'train.csv'
    __TEST_FILE_NAME = 'test.csv'

    def __init__(self, root='./data'):
        assert os.path.exists(root), '%s not exists!' % root
        self._root = os.path.expanduser(root)

        self._train_df = None
        self._test_df = None
        self._comb_df = None

    def prepare(self):
        start = time.time()
        # load train & test set
        self._train_df = pd.read_csv(os.path.join(self._root, self.__TRAIN_FILE_NAME))
        self._test_df = pd.read_csv(os.path.join(self._root, self.__TEST_FILE_NAME))
        self._comb_df = self._train_df.append(self._test_df)

        for column in self._train_df.columns:
            if self._train_df[column].dtype == object:
                self._train_df[column] = self._train_df[column].astype('category')
        for column in self._test_df.columns:
            if self._test_df[column].dtype == object:
                self._test_df[column] = self._test_df[column].astype('category')

        logging.info("prepare in %0.2fs" % (time.time() - start))

    def compute(self):
        df = self._train_df
        train_csr_matrix = coo_matrix((df['target'].astype(float),
                                       (df['msno'].cat.codes,
                                        df['song_id'].cat.codes)))
        model = implicit.als.AlternatingLeastSquares(factors=12, iterations=5)
        model.fit(train_csr_matrix)

        # generate recommendations for each user and write out to a file
        songs = dict(enumerate(df['song_id'].cat.categories))
        start = time.time()
        user_plays = train_csr_matrix.T.tocsr()
        with open('out', "w") as o:
            for userid, username in enumerate(df['msno'].cat.categories):
                for songid, score in model.recommend(userid, user_plays, N=2):
                    o.write("%s\t%s\t%s\n" % (username, songs[songid], score))
        logging.debug("generated recommendations in %0.2fs",  time.time() - start)


def main():
    fp = FeatureProducer(root='../data')
    fp.load_raw()
    fp.pre_process()
    # fp.feature_engineering()
    # fp.compute_msno_song_similarity()
    ip = ImplicitProducer(root='../data')
    ip.prepare()
    ip.compute()


if __name__ == '__main__':
    main()
