import pandas as pd
from lightfm import LightFM
DATA_DIRECTORY_PATH

train_df = pd.read_csv(DATA_DIRECTORY_PATH + 'train.csv')
test_df = pd.read_csv(DATA_DIRECTORY_PATH + 'test.csv')
comb_df = train_df.append(test_df)
members_df = pd.read_csv(DATA_DIRECTORY_PATH + 'members.csv')
songs_df = pd.read_csv(DATA_DIRECTORY_PATH + 'songs.csv')
song_extra_info_df = pd.read_csv(DATA_DIRECTORY_PATH + 'song_extra_info.csv')

songs_df = songs_df.merge(song_extra_info_df, on='song_id', how='left')

train_df = train_df.merge(songs_df, on='song_id', how='left')
test_df = test_df.merge(songs_df, on='song_id', how='left')

train_df = train_df.merge(members_df, on='msno', how='left')
test_df = test_df.merge(members_df, on='msno', how='left')

LightFM(loss='warp')