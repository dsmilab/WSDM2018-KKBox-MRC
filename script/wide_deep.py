# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import json
import pandas as pd
import itertools
import tensorflow as tf


parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='/tmp/census_model',
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='wide_deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--train_epochs', type=int, default=50, help='Number of training epochs.')

parser.add_argument(
    '--train_steps', type=int, default=2000, help='Number of training steps.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=5,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=40, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='/tmp/census_data/re_train.csv',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='/tmp/census_data/re_test.csv',
    help='Path to the test data.')

parser.add_argument(
    '--cols_data', type=str, default='/tmp/census_data/cols.json',
    help='Path to the column data.')

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous columns
    song_length = tf.feature_column.numeric_column('song_length')

    source_system_tab = tf.feature_column.categorical_column_with_vocabulary_list(
        'source_system_tab', [
            'explore', 'my library', 'search', 'discover', 'others', 'radio',
            'listen with', 'notification', 'settings'])

    source_screen_name = tf.feature_column.categorical_column_with_vocabulary_list(
        'source_screen_name', [
            'Explore', 'Local playlist more', 'others', 'My library',
            'Online playlist more', 'Album more', 'Discover Feature', 'Unknown',
            'Discover Chart', 'Radio', 'Artist more', 'Search',
            'Others profile more', 'Search Trends', 'Discover Genre',
            'My library_Search', 'Search Home', 'Discover New',
            'Self profile more', 'Concert', 'Payment', 'People local',
            'People global'])
    source_type = tf.feature_column.categorical_column_with_vocabulary_list(
        'source_type', [
            'online-playlist', 'local-playlist', 'local-library',
            'top-hits-for-artist', 'album', 'nan', 'song-based-playlist', 'radio',
            'song', 'listen-with', 'artist', 'topic-article-playlist',
            'my-daily-playlist'])

    # language = tf.feature_column.categorical_column_with_vocabulary_list(
    #     'language', [
    #         '3.0', '17.0', '31.0', '52.0', '24.0', '-1.0', '10.0', 'na_for_later', '45.0',
    #                '59.0', '38.0', , 'nan'])

    # To show an example of hashing:
    # genre_ids = tf.feature_column.categorical_column_with_hash_bucket(
    #     'genre_ids', hash_bucket_size=1000)
    #
    # artist_name = tf.feature_column.categorical_column_with_hash_bucket(
    #     'artist_name', hash_bucket_size=1000)

    # Transformations.
    # age_buckets = tf.feature_column.bucketized_column(
    #     age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns.
    base_columns = [
        song_length, source_type, source_screen_name, source_system_tab
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            ['source_type', 'source_system_tab', 'source_screen_name'], hash_bucket_size=1000),
        # tf.feature_column.crossed_column(
        #     [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        # age,
        # education_num,
        # capital_gain,
        # capital_loss,
        # hours_per_week,
        # tf.feature_column.indicator_column(workclass),
        # tf.feature_column.indicator_column(education),
        # tf.feature_column.indicator_column(marital_status),
        # tf.feature_column.indicator_column(relationship),
        # To show an example of embedding
        # tf.feature_column.embedding_column(artist_name, dimension=10),
    ]

    return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 75, 50, 25]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units,
            config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size, is_pred):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have either run data_download.py or '
        'set both arguments --train_data and --test_data.' % data_file)

    if is_pred:
        _CSV_COLUMN_DEFAULTS = _TEST_DEFAULTS
        _CSV_COLUMNS = _TEST_COLUMNS

    else:
        _CSV_COLUMN_DEFAULTS = _TRAIN_DEFAULTS
        _CSV_COLUMNS = _TRAIN_COLUMNS

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        if is_pred:
            labels = features.pop('id')
        else:
            labels = features.pop('target')

        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    if is_pred:
        return features, None
    else:
        return features, labels

def main(unused_argv):
    # Clean up the model directory if present
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        model.train(input_fn=lambda: input_fn(
            FLAGS.train_data, FLAGS.epochs_per_eval, False, FLAGS.batch_size, False), steps=FLAGS.train_steps)

        # results = model.evaluate(input_fn=lambda: input_fn(
        # FLAGS.test_data, 1, False, FLAGS.batch_size, True), steps=FLAGS.train_steps)

        # Display evaluation metrics
        # print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
        print('-' * 60)

    results = model.predict(input_fn=lambda: input_fn(
        FLAGS.test_data, 1, False, 1, True))

    for p in itertools.islice(results, 6):
        print(p['probabilities'])

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()

    cols_dict = json.load(open(FLAGS.cols_data))
    _TRAIN_COLUMNS = cols_dict['train']['cols']
    _TEST_COLUMNS = cols_dict['test']['cols']
    _TRAIN_DEFAULTS = cols_dict['train']['type']
    _TEST_DEFAULTS = cols_dict['test']['type']

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
