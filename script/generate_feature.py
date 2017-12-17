# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import pickle
import os
import json
from module.FeatureProducer import FeatureProducer
import lightgbm as lgb

LOG_FORMAT = '%(asctime)s %(levelname)s << %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt='%H:%M:%S')

pd.options.mode.chained_assignment = None  # default='warn'


def default_value(t):
    if 'object' in str(t):
        return ''
    elif 'float' in str(t):
        return 0.0
    else:
        return 0


def main():
    fp = FeatureProducer(root='../data')
    fp.load_raw()
    fp.pre_process()
    fp.feature_engineering()
    # fp.self_fit_transform()

    train_dict = {'cols': list(fp.train_df.columns), 'type': [[default_value(t)] for t in fp.train_df.dtypes]}
    test_dict = {'cols': list(fp.test_df.columns), 'type': [[default_value(t)] for t in fp.test_df.dtypes]}

    with open('feature/cols.json', 'w') as f:
        json.dump({'train': train_dict, 'test': test_dict}, f)

    fp.train_df.to_csv('feature/re_train.csv', index=False, header=False)
    fp.test_df.to_csv('feature/re_test.csv', index=False, header=False)


if __name__ == '__main__':
    main()
