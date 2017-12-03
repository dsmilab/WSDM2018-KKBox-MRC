# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import pickle
import os
import json
from utils.data import ImplicitProcesser
import lightgbm as lgb


MODEL_FILE_NAME = 'lgbm.txt'
MODEL_DIR = './model'
LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'

def save(target, name):
    pickle.dump(target, open(os.path.join(MODEL_DIR, name), 'wb'))

def main():
    logging.debug('>> Train implicit model ...')

    preprocessor = ImplicitProcesser(root='./data',
                                     feature_size=50,
                                     real_test=True,
                                     iterations=30,
                                     calculate_training_loss=True,
                                     save_dir='./model',
                                     rm_rare=False)

    X_train, y_train = preprocessor.load(train=True)
    X_test, y_id = preprocessor.load(train=False)

    if not os.path.exists(MODEL_FILE_NAME):
        logging.debug('>> Model configure not found!')
        logging.debug('>> Create model...')

        # First, no CV
        train_set = lgb.Dataset(X_train, y_train)
        valid_set = [train_set]

        params = dict({
            'learning_rate': 0.1,
            'application': 'binary',
            'min_data_in_leaf': 4,
            'max_depth': 8,
            'num_leaves': 2 ** 8,
            'verbosity': 0,
            'num_iterations': 350,
            'metric': 'auc'
        })

        model = lgb.train(params, train_set=train_set, valid_sets=valid_set, num_boost_round=100)
        # model.save_model(os.path.join(MODEL_DIR, MODEL_FILE_NAME))
        # model_json = model.dump_model(model.best_iteration)
        # json.dump(model_json, open(os.path.join(MODEL_DIR, 'model_result.json'), 'w'), indent=4)
        # feature names
        logging.debug('Feature names:', model.feature_name())

        # feature importance
        # logging.debug('Feature importance:', list(model.feature_importance()))

        logging.debug('>> Done!')

    # logging.debug('>> Load model configure...')
    # model = lgb.Booster(model_file=os.path.join(MODEL_DIR, MODEL_FILE_NAME))
    # logging.debug(model.feature_importance())
    logging.debug('>> Predicting...')
    y_test = model.predict(X_test)
    result = pd.DataFrame()
    result['id'] = y_id
    result['target'] = y_test
    result.to_csv('./submit/submission.csv', index=False)

    logging.debug('>> Done!')

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt='%H:%M:%S')
    main()

