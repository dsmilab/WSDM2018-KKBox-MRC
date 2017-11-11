import seaborn as sns
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import datetime as dt
import random
from sklearn.model_selection import GridSearchCV

import json
import os.path
import gc


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

        model = lgb.train(params, train_set=train_set, valid_sets=watchlist,
                          num_boost_round=num_boost_round, verbose_eval=5)

        tip_txt = '[GridSearch]'
        for idx in range(len(keys)):
            tip_txt += ' ' + str(keys[idx]) + '=' + str(comb[idx])
        tip_txt += ' {best_score: ' + str(model.best_score['training']['auc']) + '}'
        print(tip_txt)

        if grid_best_score is None or model.best_score['training']['auc'] > grid_best_score:
            grid_best_params, grid_best_score = own_params, model.best_score['training']['auc']

    tip_txt = '[GS Best Result]'
    for key, val in grid_best_params.items():
        tip_txt += ' ' + str(key) + '=' + str(val)
    tip_txt += ' {best_score: ' + str(grid_best_score) + '}'
    print(tip_txt)



