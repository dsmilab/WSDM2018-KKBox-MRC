import numpy as np
import pandas as pd
import time
import datetime as dt
import os
import logging
from abc import abstractmethod

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity

LOG_FORMAT = '%(asctime)s %(levelname)s << %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt='%H:%M:%S')


class DataProcessor(object):

    def __init__(self):
        return

    @abstractmethod
    def parse(self, df):
        raise NotImplementedError("Please implement method \'parse()\'.")

    @staticmethod
    def process(df, command, ref_df=None, start_index=0):
        start = time.time()

        res = None
        message = None
        if command in ['train', 'test']:
            res = TrainTestProcessor(ref_df).parse(df)
            message = command
        elif command == 'members':
            res = MembersProcessor().parse(df)
            message = command
        elif command == 'songs':
            res = SongsProcessor().parse(df)
            message = command
        elif command == 'song_extra_info':
            res = SongExtraProcessor().parse(df)
            message = command
        elif command == 'engineering':
            assert ref_df is not None, 'Please pass the reference dataframe'
            res = EngineeringProcessor(ref_df).parse(df)
            message = command
        elif command == 'timestamp':
            res = TimeStampProcessor().parse(df)
            message = command

        assert res is not None, logging.error("command \"%s\" is valid." % command)
        logging.info("parse %s_df in %0.2fs" % (message, time.time() - start))

        return res


class SongsProcessor(DataProcessor):

    def __init__(self):
        super(SongsProcessor, self).__init__()

    def parse(self, df):
        # fill missing data
        df['artist_name'].fillna('no_artist', inplace=True)
        df['language'].fillna('nan', inplace=True)
        df['composer'].fillna('nan', inplace=True)
        df['lyricist'].fillna('nan', inplace=True)
        df['genre_ids'].fillna('nan', inplace=True)

        # howeverforever
        df['artist_count'] = df['artist_name'].apply(SongsProcessor.__parse_splitted_category_to_number)
        df['genre_count'] = df['genre_ids'].apply(SongsProcessor.__parse_splitted_category_to_number)
        df['composer_count'] = df['composer'].apply(SongsProcessor.__parse_splitted_category_to_number)
        df['lyricist_count'] = df['lyricist'].apply(SongsProcessor.__parse_splitted_category_to_number)

        df['1h_lang'] = df['language'].apply(SongsProcessor.__one_hot_encode_lang)

        df['1h_song_length'] = df['song_length'].apply(lambda x: 1 if x <= 239738 else 0)

        assert(~df.isnull().any().any()), 'There exists missing data!'

        return df

    @staticmethod
    def __parse_splitted_category_to_number(x):
        if x is np.nan:
            return 0
        x = str(x)
        x.replace('/', '|')
        x.replace(';', '|')
        x.replace('\\', '|')
        x.replace(' and ', '|')
        x.replace('&', '|')
        x.replace('+', '|')
        return x.count('|') + 1

    @staticmethod
    def __one_hot_encode_lang(x):
        return 1 if x in [-1, 17, 45] else 0


class MembersProcessor(DataProcessor):

    def __init__(self):
        super(MembersProcessor, self).__init__()

    def parse(self, df):
        # fill missing data
        df['gender'].fillna('nan', inplace=True)

        # feature engineering
        df['membership_days'] = df.apply(MembersProcessor.__transform_two_dates_to_days, axis=1)

        df['registration_init_year'] = df['registration_init_time'].apply(lambda x: int(str(x)[:4]))
        df['registration_init_month'] = df['registration_init_time'].apply(lambda x: int(str(x)[4:6]))

        df['expiration_date_year'] = df['expiration_date'].apply(lambda x: int(str(x)[:4]))
        df['expiration_date_month'] = df['expiration_date'].apply(lambda x: int(str(x)[4:6]))

        # useless feature
        df.drop(['registration_init_time'], axis=1, inplace=True)

        # howeverforever
        df['bd'] = df['bd'].apply(MembersProcessor.__transform_bd_outliers)
        df['1h_via'] = df['registered_via'].apply(MembersProcessor.__one_hot_encode_via)

        assert (~df.isnull().any().any()), 'There exists missing data!'

        return df

    @staticmethod
    def __transform_bd_outliers(bd):
        # figure is from "exploration"
        if bd >= 120 or bd <= 7:
            return 'nan'
        mean = 28.99737187910644
        std = 9.538470787507382
        return bd if abs(bd - mean) <= 3 * std else 'nan'

    @staticmethod
    def __one_hot_encode_via(x):
        return 0 if x == 4 else 1

    @staticmethod
    def __parse_str_to_date(date_str):
        # [format] yyyymmdd
        date_str = str(date_str)
        assert (isinstance(date_str, str))
        assert (len(date_str) == 8)

        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:])
        return dt.date(year, month, day)

    @staticmethod
    def __transform_two_dates_to_days(row):
        start = MembersProcessor.__parse_str_to_date(row['registration_init_time'])
        end = MembersProcessor.__parse_str_to_date(row['expiration_date'])
        delta = end - start
        return delta.days


class SongExtraProcessor(DataProcessor):

    def __init__(self):
        super(SongExtraProcessor, self).__init__()

    def parse(self, df):
        df['song_year'] = df['isrc'].apply(SongExtraProcessor.__transform_isrc_to_year)
        df.drop(['name', 'isrc'], axis=1, inplace=True)

        # howeverforever
        # df['song_country'] = df['isrc'].apply(self._transform_isrc_to_country)
        # df['song_registration'] = df['isrc'].apply(self._transform_isrc_to_reg)
        # df['song_designation'] = df['isrc'].apply(self._transform_isrc_to_desig)

        df['1h_song_year'] = df['song_year'].apply(SongExtraProcessor.__one_hot_encode_year)
        # df['1h_song_country'] = df['song_country'].apply(self._one_hot_encode_country)

        df['song_year'].fillna(2017, inplace=True)
        # df['song_registration'].fillna('***', inplace=True)

        assert (~df.isnull().any().any())

        return df

    @staticmethod
    def __transform_isrc_to_year(isrc):
        if type(isrc) != str:
            return np.nan
        # this year 2017
        suffix = int(isrc[5:7])
        return 1900 + suffix if suffix > 17 else 2000 + suffix

    @staticmethod
    def __one_hot_encode_year(x):
        return 1 if 2013 <= float(x) <= 2017 else 0


class TrainTestProcessor(DataProcessor):

    def __init__(self, ref_df):
        super(TrainTestProcessor, self).__init__()
        self._ref_df = ref_df.copy()

        # fill missing data
        self._ref_df['source_system_tab'].fillna('others', inplace=True)
        self._ref_df['source_screen_name'].fillna('others', inplace=True)
        self._ref_df['source_type'].fillna('nan', inplace=True)

        # feature engineering
        self._ref_df['source_merged'] = self._ref_df['source_system_tab'].map(str) + ' | ' + \
                                        self._ref_df['source_screen_name'].map(str) + ' | ' + \
                                        self._ref_df['source_type'].map(str)

        self._ref_df = self._ref_df[['source_merged', 'target']].groupby('source_merged').agg(['mean', 'count'])
        self._ref_df.reset_index(inplace=True)
        self._ref_df.columns = ['source_merged', 'source_replay_pb', 'source_replay_count']

    def parse(self, df):
        # fill missing data
        df['source_system_tab'].fillna('others', inplace=True)
        df['source_screen_name'].fillna('others', inplace=True)
        df['source_type'].fillna('nan', inplace=True)

        # feature engineering
        df['source_merged'] = df['source_system_tab'].map(str) + ' | ' +\
                              df['source_screen_name'].map(str) + ' | ' +\
                              df['source_type'].map(str)

        df = df.merge(self._ref_df, on='source_merged', how='left')

        df['1h_source'] = df['source_replay_pb'].apply(TrainTestProcessor.__one_hot_encode_source)

        df['1h_system_tab'] = df['source_system_tab'].apply(TrainTestProcessor.__one_hot_encode_system_tab)
        df['1h_screen_name'] = df['source_screen_name'].apply(TrainTestProcessor.__one_hot_encode_screen_name)
        df['1h_source_type'] = df['source_type'].apply(TrainTestProcessor.__one_hot_encode_source_type)

        # useless feature
        df.drop(['source_merged', 'source_replay_pb', 'source_replay_count'], axis=1, inplace=True)

        assert (~df.isnull().any().any()), 'There exists missing data!'

        return df

    @staticmethod
    def __one_hot_encode_system_tab(x):
        return 1 if x == 'my library' else 0

    @staticmethod
    def __one_hot_encode_screen_name(x):
        return 1 if x == 'Local playlist more' or x == 'My library' else 0

    @staticmethod
    def __one_hot_encode_source_type(x):
        return 1 if x == 'local-library' or x == 'local-playlist' else 0

    @staticmethod
    def __one_hot_encode_source(x):
        return 1 if x >= 0.6 else 0


class EngineeringProcessor(DataProcessor):

    def __init__(self, ref_df):
        super(EngineeringProcessor, self).__init__()
        self._ref_df = ref_df

    def parse(self, df):
        df = self.generate_play_count(df)
        df = self.generate_track_count(df)
        df = self.generate_cover_lang(df)

        return df

    def generate_play_count(self, df):
        count_df = self._ref_df['song_id'].value_counts().reset_index()
        count_df.columns = ['song_id', 'play_count']

        df = df.merge(count_df, on='song_id', how='left')
        df['play_count'].fillna(0, inplace=True)

        return df

    def generate_track_count(self, df):
        track_count_df = self._ref_df[['song_id', 'artist_name']].drop_duplicates('song_id')
        track_count_df = track_count_df.groupby('artist_name').agg('count').reset_index()
        track_count_df.columns = ['artist_name', 'track_count']
        track_count_df = track_count_df.sort_values('track_count', ascending=False)

        artist_count_df = self._ref_df[['artist_name', 'target']].groupby('artist_name').agg(['mean', 'count']).reset_index()
        artist_count_df.columns = ['artist_name', 'replay_pb', 'play_count']

        artist_count_df = artist_count_df.merge(track_count_df, on='artist_name', how='left')

        df = df.merge(artist_count_df[['artist_name', 'track_count']], on='artist_name', how='left')
        df['track_count'].fillna(0, inplace=True)

        return df

    def generate_cover_lang(self, df):
        cover_lang_df = self._ref_df[['artist_name', 'language']].drop_duplicates(['artist_name', 'language'])
        cover_lang_df = cover_lang_df['artist_name'].value_counts().reset_index()
        cover_lang_df.columns = ['artist_name', 'cover_lang']

        df = df.merge(cover_lang_df, on='artist_name', how='left')
        df['cover_lang'].fillna(0, inplace=True)

        return df


class TimeStampProcessor(DataProcessor):
    """
    This is order-sensitive.
    """
    __TIMESTAMP_COLUMN_NAME = 'timestamp'

    def __init__(self):
        super(TimeStampProcessor, self).__init__()
        self._now_index = 0

    def parse(self, df):
        timestamp_df = pd.DataFrame(data=np.arange(self._now_index, self._now_index + df.shape[0]),
                                    columns=[TimeStampProcessor.__TIMESTAMP_COLUMN_NAME])
        self._now_index += df.shape[0]
        df = df.join(timestamp_df)

        return df


class SimilarityProcessor(DataProcessor):
    __MEMBERS_FEATURE = ['city', 'bd', 'gender', 'registered_via', 'expiration_date', 'membership_days',
                         'registration_init_year', 'registration_init_month', 'registration_init_time',
                         'expiration_date_year', 'expiration_date_month']

    __SONGS_FEATURE = ['genre_ids', 'artist_name', 'language', 'composer', 'lyricist', 'song_year']

    def __init__(self, songs_df, members_df):
        super(SimilarityProcessor, self).__init__()
        self._songs_df = songs_df
        self._members_df = members_df

    def parse(self, df):
        train_df = df[0]
        test_df = df[1]

        return self.__compute_msno_song_similarity(train_df, test_df)

    class ColumnSelector(BaseEstimator, TransformerMixin):

        def __init__(self, columns):
            self.columns = columns

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X[self.columns].to_dict(orient='record')

    def __compute_msno_song_similarity(self, train, test):
        for col in train.columns:
            if train[col].dtype == object:
                train[col] = train[col].astype('category')
                test[col] = test[col].astype('category')
        # pipeline
        member_pipeline = Pipeline([
                ('extract', SimilarityProcessor.ColumnSelector(SimilarityProcessor.__MEMBERS_FEATURE)),
                ('dicVect', DictVectorizer())])
        song_pipeline = Pipeline([
                ('extract', SimilarityProcessor.ColumnSelector(SimilarityProcessor.__SONGS_FEATURE)),
                ('dicVect', DictVectorizer())])

        # ? songs = self.songs.merge(self.extra, on='song_id', how='left').fillna('test')
        songs_df = self._songs_df
        members_df = self._members_df.fillna('test')
        msno_x = {v: i for i, v in enumerate(members_df.msno)}
        song_x = {v: i for i, v in enumerate(songs_df.song_id)}

        # transform members_df
        start = time.time()
        msno_m = member_pipeline.fit_transform(members_df)
        logging.debug("transform members_df in %0.2fs" % (time.time() - start))

        # transform songs_df
        start = time.time()
        song_m = song_pipeline.fit_transform(songs_df)
        logging.debug("transform songs_df in %0.2fs" % (time.time() - start))

        known_msno = set(train.msno.unique())
        unknown_msno = list(set(test.msno.unique()) - known_msno)
        total_msno = float(len(unknown_msno))

        known_song = set(train.song_id.unique())
        unknown_song = list(set(test.song_id.unique()) - known_song)
        total_song = float(len(unknown_song))

        unknown_msno_map, unknown_song_map = {}, {}

        start = time.time()
        known_msno_list = members_df.msno.apply(lambda x: x in known_msno)
        known_song_list = songs_df.song_id.apply(lambda x: x in known_song)
        logging.debug("establish known list in %0.2fs" % (time.time() - start))

        # ? Parallel(n_jobs=6)(delayed(self._get_unknown_map)
        #                    (i, members.msno, known_msno_list, True) for i in unknown_msno)

        start = time.time()
        n = 0
        for i in unknown_msno:
            if i in msno_x:
                df = SimilarityProcessor.__get_rank(msno_m, msno_x[i], members_df.msno, known_msno_list)
                unknown_msno_map[i] = df.iloc[0]['id']
            else:
                unknown_msno_map[i] = 'new'
            n += 1
            if (n + 1) % 100 == 0:
                print('msno: %f %%' % ((n/total_msno) * 100))

        n = 0
        for i in unknown_song:
            if i in song_x:
                df = SimilarityProcessor.__get_rank(song_m, song_x[i], songs_df.song_id, known_song_list)
                unknown_song_map[i] = df.iloc[0]['id']
            else:
                unknown_song_map[i] = 'new'
            n += 1
            if (n + 1) % 100 == 0:
                print('song: %f %%' % ((n/total_song) * 100))

        logging.debug("transform all unknown data in %0.2fs" % (time.time() - start))
        return unknown_msno_map, unknown_song_map

    @staticmethod
    def __get_rank(model, w, id_list, known_list):
        result = cosine_similarity(model, model[w].toarray().reshape(1, -1)).reshape(1, -1)[0]
        r = pd.DataFrame({'id': id_list, 'similarity': result, 'known': known_list})
        return r[r.known].sort_values(by='similarity', ascending=False).reset_index(drop=True)


class FeatureProducer(object):

    __SONGS_FILE_NAME = 'songs.csv'
    __SONG_EXTRA_FILE_NAME = 'song_extra_info.csv'
    __MEMBERS_FILE_NAME = 'members.csv'
    __TRAIN_FILE_NAME = 'train.csv'
    __TEST_FILE_NAME = 'test.csv'

    __INITIALIZATION_READY = (1 << 0)
    __LOAD_READY = (1 << 1)
    __PREPROCESS_READY = (1 << 2)
    __ENGINEERING_READY = (1 << 3)
    __SIMILARITY_MAPPING_READY = (1 << 4)

    __FINAL_TYPE_TABLE = {
        'msno': 'category',
        'song_id': 'category',
        'source_system_tab': 'category',
        'source_screen_name': 'category',
        'source_type': 'category',
        '1h_source': bool,
        '1h_system_tab': bool,
        '1h_screen_name': bool,
        '1h_source_type': bool,
        'song_length': np.int32,
        'genre_ids': 'category',
        'artist_name': 'category',
        'composer': 'category',
        'lyricist': 'category',
        'language': 'category',
        'artist_count': np.int32,
        'genre_count': np.int32,
        'composer_count': np.int32,
        'lyricist_count': np.int32,
        '1h_lang': bool,
        '1h_song_length': bool,
        'song_year': np.int32,
        '1h_song_year': bool,
        'city': 'category',
        'bd': 'category',
        'gender': 'category',
        'registered_via': 'category',
        'expiration_date': np.int32,
        'membership_days': np.int32,
        'registration_init_year': np.int32,
        'registration_init_month': np.int32,
        'expiration_date_year': np.int32,
        'expiration_date_month': np.int32,
        '1h_via': bool,
        'play_count': np.int32,
        'track_count': np.int32,
        'cover_lang': np.int32
    }

    def __init__(self, root='./data'):
        assert os.path.exists(root), '%s not exists!' % root
        self._root = os.path.expanduser(root)

        self._songs_df = None
        self._song_extra_info_df = None
        self._members_df = None
        self._train_df = None
        self._test_df = None
        self._comb_df = None
        self._unknown_msno_map = None
        self._unknown_song_map = None
        self._state = FeatureProducer.__INITIALIZATION_READY

    def load_raw(self):
        """
        Load all raw data under the directory specified.
        Call this function right after initialization.

        :return:
        """

        assert (self._state & FeatureProducer.__INITIALIZATION_READY) > 0, logging.error("Please reconstruct new class")

        start = time.time()

        # load train & test set
        self._train_df = pd.read_csv(os.path.join(self._root, self.__TRAIN_FILE_NAME))
        self._test_df = pd.read_csv(os.path.join(self._root, self.__TEST_FILE_NAME))
        # load song & member set
        self._songs_df = pd.read_csv(os.path.join(self._root, self.__SONGS_FILE_NAME))
        self._song_extra_info_df = pd.read_csv(os.path.join(self._root, self.__SONG_EXTRA_FILE_NAME))
        self._members_df = pd.read_csv(os.path.join(self._root, self.__MEMBERS_FILE_NAME))

        self._state |= FeatureProducer.__LOAD_READY
        logging.info("load raw data in %0.2fs" % (time.time() - start))

    def pre_process(self):
        """
        Pre-process all dataframes and merge them into "train_df" and "test_df".
        Call this function after calling "load_raw"

        :return:
        """

        assert (self._state & FeatureProducer.__LOAD_READY) > 0, logging.error("Please load raw data first")

        # pre-process all data-frame
        self._train_df = DataProcessor().process(self._train_df, 'train', self._train_df)
        self._test_df = DataProcessor().process(self._test_df, 'test', self._train_df)
        self._members_df = DataProcessor().process(self._members_df, 'members')
        self._songs_df = DataProcessor().process(self._songs_df, "songs")
        self._song_extra_info_df = DataProcessor().process(self._song_extra_info_df, "song_extra_info")

        # merge all data-frame
        self._songs_df = self._songs_df.merge(self._song_extra_info_df, on='song_id', how='left')

        self._train_df = self._train_df.merge(self._songs_df, on='song_id', how='left')
        self._test_df = self._test_df.merge(self._songs_df, on='song_id', how='left')

        self._train_df = self._train_df.merge(self._members_df, on='msno', how='left')
        self._test_df = self._test_df.merge(self._members_df, on='msno', how='left')

        self._comb_df = self._train_df.append(self._test_df)

        self._state |= FeatureProducer.__PREPROCESS_READY

    def feature_engineering(self):
        """
        Do the advanced feature engineering.
        Call this function after calling "pre_process"

        :return:
        """

        assert (self._state & FeatureProducer.__PREPROCESS_READY) > 0, logging.error("Please proprocess raw data first")

        self._train_df = DataProcessor().process(self._train_df, 'engineering', self._train_df)
        self._test_df = DataProcessor().process(self._test_df, 'engineering', self._comb_df)

        tsp = TimeStampProcessor()
        self._train_df = tsp.parse(self._train_df)
        self._test_df = tsp.parse(self._test_df)

        self._state |= FeatureProducer.__ENGINEERING_READY

    def self_fit_transform(self):

        self._train_df.fillna(0, inplace=True)
        self._test_df.fillna(0, inplace=True)

        for column, dtype in self.__FINAL_TYPE_TABLE.items():
            start = time.time()

            self._train_df[column] = self._train_df[column].astype(dtype)
            self._test_df[column] = self._test_df[column].astype(dtype)

            logging.info("transform \"%s\" in %0.2fs" % (column, time.time() - start))

        for column in self._train_df.columns:
            if self._train_df[column].dtype == 'object':
                self._train_df[column] = self._train_df[column].astype('category')
        for column in self._test_df.columns:
            if self._test_df[column].dtype == 'object':
                self._test_df[column] = self._test_df[column].astype('category')

    def compute_msno_song_similarity(self):
        """
        I don't really know about how this do...
        Call this function after calling "load_raw"

        :return:
        """

        assert (self._state & FeatureProducer.__LOAD_READY) > 0, logging.error("Please load raw data first")

        self._unknown_msno_map, self._unknown_song_map = \
            SimilarityProcessor(self._songs_df, self._members_df).parse([self._train_df, self._test_df])

        self._state |= FeatureProducer.__SIMILARITY_MAPPING_READY

    @property
    def train_df(self):
        return self._train_df

    @property
    def test_df(self):
        return self._test_df

    @property
    def members_df(self):
        return self._members_df

    @property
    def songs_df(self):
        return self._songs_df

    @property
    def song_extra_info_df(self):
        return self._song_extra_info_df

    @property
    def comb_df(self):
        return self._comb_df
