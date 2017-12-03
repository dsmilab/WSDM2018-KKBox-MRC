import pandas as pd
import numpy as np
import lightgbm as lgb

import json
import os.path
import gc

MODEL_FILE_NAME = 'model.txt'


def transform_isrc_to_year(isrc):
    if type(isrc) != str:
        return np.nan
    # this year 2017
    suffix = int(isrc[5:7])
    return 1900 + suffix if suffix > 17 else 2000 + suffix


def main():
    print('>> Load CSV data...')
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    songs_df = pd.read_csv('data/songs.csv')
    song_extra_info_df = pd.read_csv('data/song_extra_info.csv')
    members_df = pd.read_csv('data/members.csv')
    print('>> Merge needed information...')
    song_extra_info_df['song_year'] = song_extra_info_df['isrc'].apply(transform_isrc_to_year)
    song_extra_info_df.drop(['name', 'isrc'], axis=1, inplace=True)

    songs_df = songs_df.merge(song_extra_info_df, on='song_id', how='left')

    train_df = train_df.merge(songs_df, on='song_id', how='left')
    test_df = test_df.merge(songs_df, on='song_id', how='left')

    train_df = train_df.merge(members_df, on='msno', how='left')
    test_df = test_df.merge(members_df, on='msno', how='left')

    del members_df, song_extra_info_df
    gc.collect()

    for column in train_df.columns:
        if train_df[column].dtype == object:
            train_df[column] = train_df[column].astype('category')
    for column in test_df.columns:
        if test_df[column].dtype == object:
            test_df[column] = test_df[column].astype('category')

    x = train_df.drop(['target'], axis=1)
    y = train_df['target'].values

    x_test = test_df.drop(['id'], axis=1)
    test_ids = test_df['id'].values

    if not os.path.exists('model.txt'):
        print('>> >> model configure not found!')
        print('>> Create model...')

        # First, no CV
        train_set = lgb.Dataset(x, y)
        valid_set = [train_set]

        params = dict({
            'learning_rate': 0.1,
            'application': 'binary',
            'min_data_in_leaf': 4,
            'max_depth': 8,
            'num_leaves': 2 ** 8,
            'verbosity': 0,
            'metric': 'auc'
        })

        model = lgb.train(params, train_set=train_set, valid_sets=valid_set, num_boost_round=100)
        model.save_model(MODEL_FILE_NAME)
        model_json = model.dump_model(model.best_iteration)
        json.dump(model_json, open('model_result.json', 'w'), indent=4)
        # feature names
        print('Feature names:', model.feature_name())

        # feature importance
        print('Feature importance:', list(model.feature_importance()))

        print('>> >> Done!')

    print('>> Load model configure...')
    model = lgb.Booster(model_file=MODEL_FILE_NAME)
    print(model.feature_importance())
    print('>> Predicting...')
    y_test = model.predict(x_test)
    submission_df = pd.DataFrame()
    submission_df['id'] = test_ids
    submission_df['target'] = y_test
    submission_df.to_csv('data/submission.csv', index=False)

    print('>> Done!')


if __name__ == '__main__':
    main()
