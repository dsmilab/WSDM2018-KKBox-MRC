import implicit
import time
import os
import logging

# self-defined
import seaborn as sns
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import time

import os.path

from scipy.sparse import coo_matrix

from module.FeatureProducer import FeatureProducer
MODEL_FILE_NAME = 'model.txt'

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


def custom_cv(params, train_set, hold_out_set=None, k_fold=5, num_boost_round=20):
    x_train = train_set.data
    y_train = train_set.label
    n = x_train.shape[0]
    unit = n // k_fold

    cv_scores = []
    for k in range(k_fold):
        ##### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if k < k_fold - 1:
            continue
        ##### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x_cv_valid = None
        y_cv_valid = None
        if k == k_fold - 1:
            x_cv_valid = x_train[unit * k:]
            y_cv_valid = y_train[unit * k:]
        else:
            x_cv_valid = x_train[unit * k: unit * (k + 1)]
            y_cv_valid = y_train[unit * k: unit * (k + 1)]

        x_cv_train = None
        y_cv_train = None
        if k == 0:
            x_cv_train = x_train[unit * (k + 1):]
            y_cv_train = y_train[unit * (k + 1):]
        elif k == k_fold - 1:
            x_cv_train = x_train[:unit * k]
            y_cv_train = y_train[:unit * k]
        else:
            x_cv_train = x_train[:unit * k].append(x_train[unit * (k + 1):])
            y_cv_train = y_train[:unit * k].append(y_train[unit * (k + 1):])

        cv_train_set = lgb.Dataset(x_cv_train, y_cv_train)
        cv_valid_set = lgb.Dataset(x_cv_valid, y_cv_valid)
        watchlist = [cv_valid_set]

        # not tested yet
        if hold_out_set is not None:
            watchlist.append(hold_out_set)
        model = lgb.train(params, train_set=cv_train_set, valid_sets=watchlist,
                          num_boost_round=num_boost_round, verbose_eval=5)
        print(model.best_score)
        cv_scores.append(model.best_score['valid_1']['auc'])

    tip_txt = '[CV]'
    tip_txt += ' ' + str(cv_scores)
    mean_cv_score = np.mean(cv_scores)
    tip_txt += '{ auc score=' + str(mean_cv_score) + ' }'
    print(tip_txt)

    return mean_cv_score


def custom_grid_search(params, own_grid_params, train_set, valid_set, num_boost_round=20):
    keys = []
    values = [list()]
    for key, value in own_grid_params.items():
        keys.append(key)
        new_values = []
        for item in values:
            for val in value:
                new_values.append(item + [val])
        values = new_values

    watchlist = [valid_set]
    grid_best_params = None
    grid_best_score = None

    for comb in values:
        own_params = {}
        for idx in range(len(keys)):
            own_params[keys[idx]] = comb[idx]
            params[keys[idx]] = comb[idx]

        cv_score = custom_cv(params, train_set, valid_set, k_fold=4, num_boost_round=num_boost_round)

        tip_txt = '[GridSearch]'
        for idx in range(len(keys)):
            tip_txt += ' ' + str(keys[idx]) + '=' + str(comb[idx])
        tip_txt += ' { best_score: ' + str(cv_score) + ' }'
        print(tip_txt)

        if grid_best_score is None or cv_score > grid_best_score:
            grid_best_params, grid_best_score = own_params, cv_score

    tip_txt = '[GS Best Result]'
    for key, val in grid_best_params.items():
        tip_txt += ' ' + str(key) + '=' + str(val)
    tip_txt += ' { best_score: ' + str(grid_best_score) + ' }'
    print(tip_txt)

    return grid_best_params


def main():
    fp = FeatureProducer(root='../data')
    fp.load_raw()
    fp.pre_process()
    fp.feature_engineering()
    fp.self_fit_transform()
    # fp.compute_msno_song_similarity()
    # ip = ImplicitProducer(root='../data')
    # ip.prepare()
    # ip.compute()
    train_df = fp.train_df
    test_df = fp.test_df

    assert (train_df.shape[1] == test_df.shape[1])
    train_df.info()
    test_df.info()

    x = train_df.drop(['target'], axis=1)
    y = train_df['target']

    # take the last # rows of train_df as valid set where # means number of rows in test_df
    x_valid = train_df.drop(['target'], axis=1).tail(test_df.shape[0])
    y_valid = train_df['target'].tail(test_df.shape[0])

    x_test = test_df.drop(['id'], axis=1)
    test_ids = test_df['id']

    train_df.head(15)

    train_set = lgb.Dataset(x, y)
    valid_set = lgb.Dataset(x_valid, y_valid, free_raw_data=False)
    watchlist = [valid_set]

    params = dict({
        'learning_rate': 0.2,
        'application': 'binary',
        'min_data_in_leaf': 10,
        #    'max_depth': 10,
        'num_leaves': 2 ** 7,
        'max_bin': 255,
        'verbosity': 0,
        'metric': 'auc'
    })

    grid_params = {
        'learning_rate': [0.1, 0.2],
        'max_depth': [8, 10],
    }

    # best_grid_params = custom_grid_search(params, grid_params, train_set, hold_out_set, num_boost_round=20)
    # for key, val in best_grid_params.items():
    #     params[key] = best_grid_params[key]

    cv_score = custom_cv(params, train_set, valid_set, k_fold=4, num_boost_round=100)
    print(cv_score)

    model = lgb.train(params, train_set=train_set, valid_sets=watchlist, num_boost_round=100, verbose_eval=5)
    y_test = model.predict(x_test)

    # When CV, valid_0 means the front 75% training, the last 25% validating
    #          valid_1 means the front 75% training, the last "len(test set)" validating
    #
    # When LGBM running, valid_0 means 100% training, the last "len(test set)" validating.
    #
    # Use CV valid_0 auc score to predict result!

    plot_df = pd.DataFrame({'features': train_df.columns[train_df.columns != 'target'],
                            'importance': model.feature_importance()})
    plot_df = plot_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(8, 15))
    sns.barplot(x=plot_df.importances, y=plot_df.features)
    plt.savefig('feature_importance.png')
    plt.show()

    submission_df = pd.DataFrame()
    submission_df['id'] = test_ids
    submission_df['target'] = y_test
    # string file compression reduces file size
    submission_df.to_csv('data/submission.csv.gz', compression='gzip', index=False, float_format='%.5f')
    submission_df.info()


if __name__ == '__main__':
    main()
