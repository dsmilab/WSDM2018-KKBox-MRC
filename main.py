import pandas as pd
import numpy as np
import lightgbm as lgb

import gc


def transform_isrc_to_year(isrc):
    if type(isrc) != str:
        return np.nan
    # this year 2017
    suffix = int(isrc[5:7])
    return 1900 + suffix if suffix > 17 else 2000 + suffix


def main():
    print('>> Load CSV data...')
    train_df = pd.read_csv('data/train.csv', dtype={
        'msno': 'category',
        'song_id': 'category',
        'source_system_tab': 'category',
        'source_screen_name': 'category',
        'source_type': 'category',
        'target': np.int32
    })
    test_df = pd.read_csv('data/test.csv', dtype={
        'msno': 'category',
        'song_id': 'category',
        'source_system_tab': 'category',
        'source_screen_name': 'category',
        'source_type': 'category',
        'id': np.int32
    })
    songs_df = pd.read_csv('data/songs.csv', dtype={
        'genre_ids': 'category',
        'language': 'category',
        'artist_name': 'category',
        'composer': 'category',
        'lyricist': 'category',
        'song_id': 'category',
        'song_length': 'category',
    })
    song_extra_info_df = pd.read_csv('data/song_extra_info.csv', dtype={
        'song_id': 'category',
        'name': 'category',
        'isrc': 'category'
    })
    members_df = pd.read_csv('data/members.csv', dtype={
        'msno': 'category',
        'city': 'category',
        'bd': np.int32,
        'gender': 'category',
        'registered_via': 'category',
        'registration_init_time': 'category',
        'expiration_date': 'category'
    })
    # sample_submission_df = pd.read_csv('data/sample_submission.csv')

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
            test_df[column] = test_df[column].astype('category')

    x = train_df.drop(['target'], axis=1)
    y = train_df['target'].values

    x_test = test_df.drop(['id'], axis=1)
    test_ids = test_df['id'].values

    print('>> Create model...')

    # First, no CV
    train_set = lgb.Dataset(x, y)
    valid_set = [train_set]

    params = dict({
        'learning_rate': 0.1,
        'application': 'binary',
        'max_depth': 8,
        'num_leaves': 2 ** 8,
        'verbosity': 0,
        'metric': 'auc',
        # 'device': 'gpu'
    })

    model = lgb.train(params, train_set=train_set, valid_sets=valid_set, num_boost_round=20)

    print('>> Predicting...')

    y_test = model.predict(x_test)
    submission_df = pd.DataFrame()
    submission_df['id'] = test_ids
    submission_df['target'] = y_test
    submission_df.to_csv('data/submission.csv', index=False)
    print(submission_df)

    print('>> Done!')

if __name__ == '__main__':
    main()
