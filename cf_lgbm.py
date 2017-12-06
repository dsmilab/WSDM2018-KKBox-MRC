# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import pickle
import os
import json
from utils.data import ImplicitProcesser, FeatureProcesser
import lightgbm as lgb


pd.options.mode.chained_assignment = None  # default='warn'

LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'
params = [{
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.3 ,
        'verbose': 0,
        'num_leaves': 108,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': 30,
        'num_rounds': 200,
        'metric' : 'auc'
    },{
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'dart',
        'learning_rate': 0.3 ,
        'verbose': 0,
        'num_leaves': 108,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': 20,
        'num_rounds': 200,
        'metric' : 'auc'
    }]

def main():
    logging.debug('>> Get features ...')

    feature_processer = FeatureProcesser(root='./data')
    cf_processer = ImplicitProcesser(feature_size=50,
                                     iterations=30,
                                     calculate_training_loss=True,
                                     save_dir='./model',
                                     random_state=50,
                                     n_clusters=50,
                                     cluster=True)

    train, test, unknown_msno_map, unknown_song_map = feature_processer.load()
    # train.to_csv('train.csv', index=False)
    # test.to_csv('test.csv', index=False)

    X_train, y_train, X_test, ids = cf_processer.fit(train_df=train, test_df=test, unknown_msno_map=unknown_msno_map, unknown_song_map=unknown_song_map)

    d_train_final = lgb.Dataset(X_train, y_train)
    watchlist_final = lgb.Dataset(X_train, y_train)

    model_f1 = lgb.train(params[0], train_set=d_train_final,  valid_sets=watchlist_final, verbose_eval=5)
    model_f2 = lgb.train(params[1], train_set=d_train_final,  valid_sets=watchlist_final, verbose_eval=5)

    print('Making predictions')
    p_test_1 = model_f1.predict(X_test)
    p_test_2 = model_f2.predict(X_test)
    p_test_avg = np.mean([p_test_1, p_test_2], axis = 0)

    print('Done making predictions')

    print ('Saving predictions Model model of gbdt')

    subm = pd.DataFrame()
    subm['id'] = ids
    subm['target'] = p_test_avg
    # subm['target'] = p_test_1
    subm.to_csv('./submit/submission_lgbm_avg.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

    print('Done!')

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt='%H:%M:%S')
    main()

