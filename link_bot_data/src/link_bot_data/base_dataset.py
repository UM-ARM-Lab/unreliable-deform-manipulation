#!/usr/bin/env python
from __future__ import division, print_function

import json
import pathlib
from typing import List, Optional

import tensorflow as tf
from colorama import Fore

from link_bot_data.link_bot_dataset_utils import balance_dataset, parse_and_deserialize


class BaseDataset:

    def __init__(self, dataset_dirs: List[pathlib.Path]):
        self.dataset_dirs = dataset_dirs
        self.hparams = {}
        for dataset_dir in dataset_dirs:
            dataset_hparams_filename = dataset_dir / 'hparams.json'

            # to merge dataset hparams
            hparams = json.load(open(str(dataset_hparams_filename), 'r'))
            for k, v in hparams.items():
                if k not in self.hparams:
                    self.hparams[k] = v
                elif self.hparams[k] == v:
                    pass
                elif k == 'sequence_length':
                    # always use the minimum of different sequence lengths
                    self.hparams[k] = min(self.hparams[k], hparams[k])
                else:
                    msg = "Datasets have differing values for the hparam {}, using value {}".format(k, self.hparams[k])
                    print(Fore.RED + msg + Fore.RESET)

        self.max_sequence_length = None
        self.state_like_names_and_shapes = {}
        self.action_like_names_and_shapes = {}
        self.trajectory_constant_names_and_shapes = {}

    def get_datasets(self,
                     mode: str,
                     batch_size: Optional[int],
                     seed: int,
                     shuffle: bool = True,
                     sequence_length: Optional[int] = None,
                     n_parallel_calls: int = tf.data.experimental.AUTOTUNE,
                     balance_key: Optional[str] = None) -> tf.data.Dataset:
        records = []
        for dataset_dir in self.dataset_dirs:
            records.extend(str(filename) for filename in (dataset_dir / mode).glob("*.tfrecords"))
        return self.get_datasets_from_records(records,
                                              batch_size=batch_size,
                                              seed=seed,
                                              shuffle=shuffle,
                                              sequence_length=sequence_length,
                                              balance_key=balance_key,
                                              n_parallel_calls=n_parallel_calls)

    def get_datasets_all_modes(self,
                               batch_size: Optional[int],
                               shuffle: bool = True,
                               seed: int = 0,
                               balance_key: Optional[str] = None):
        train_filenames = []
        test_filenames = []
        val_filenames = []
        for dataset_dir in self.dataset_dirs:
            train_filenames.extend(str(filename) for filename in dataset_dir.glob("{}/*.tfrecords".format('train')))
            test_filenames.extend(str(filename) for filename in dataset_dir.glob("{}/*.tfrecords".format('test')))
            val_filenames.extend(str(filename) for filename in dataset_dir.glob("{}/*.tfrecords".format('val')))

        all_filenames = train_filenames
        all_filenames.extend(test_filenames)
        all_filenames.extend(val_filenames)

        return self.get_datasets_from_records(records=all_filenames,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              seed=seed,
                                              balance_key=balance_key)

    def get_datasets_from_records(self,
                                  records: List[str],
                                  batch_size: int,
                                  seed: int,
                                  shuffle: bool = True,
                                  sequence_length: Optional[int] = None,
                                  balance_key: str = None,
                                  n_parallel_calls: int = None,
                                  ) -> tf.data.Dataset:
        self.max_sequence_length = self.hparams['sequence_length']
        if sequence_length is None:
            # set sequence_length to the longest possible if it is not specified
            sequence_length = self.max_sequence_length
            msg = "sequence length not specified, assuming hparams sequence length: {}".format(sequence_length)
            print(Fore.YELLOW + msg + Fore.RESET)

        dataset = tf.data.TFRecordDataset(records, buffer_size=1 * 1024 * 1024,
                                          compression_type=self.hparams['compression_type'])

        if shuffle:
            # TODO: tune buffer size for performance
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)

        features_description = self.make_features_description()

        dataset = parse_and_deserialize(dataset, feature_description=features_description, n_parallel_calls=n_parallel_calls)
        dataset = dataset.map(self.split_into_sequences, num_parallel_calls=n_parallel_calls)

        def _slice_sequences(constant_data, state_like_seqs, action_like_seqs):
            return self.slice_sequences(constant_data, state_like_seqs, action_like_seqs, sequence_length=sequence_length)

        dataset = dataset.map(_slice_sequences, num_parallel_calls=n_parallel_calls)

        dataset = self.post_process(dataset, n_parallel_calls)

        if balance_key is not None:
            dataset = balance_dataset(dataset, balance_key)

        if batch_size is not None:
            dataset = dataset.batch(batch_size, drop_remainder=False)

        dataset = dataset.cache()

        # sanity check that the dataset isn't empty, which can happen when debugging if batch size is bigger than dataset size
        empty = True
        for _ in dataset:
            empty = False
            break
        if empty:
            dataset_size = 0
            for _ in dataset:
                dataset_size += batch_size
                if dataset_size >= 8 * batch_size:
                    dataset_size = "... more than 8 batches"
                    break
            raise RuntimeError("Dataset is empty! batch size: {}, dataset size: {}".format(batch_size, dataset_size))

        return dataset

    @staticmethod
    def slice_sequences(constant_data, state_like_seqs, action_like_seqs, sequence_length: int):
        t_start = 0
        state_like_t_slice = slice(t_start, t_start + sequence_length)
        action_like_t_slice = slice(t_start, t_start + sequence_length - 1)

        state_like_seqs_sliced = {}
        action_like_seqs_sliced = {}
        for example_name, seq in state_like_seqs.items():
            sliced_seq = seq[state_like_t_slice]
            sliced_seq.set_shape([sequence_length] + seq.shape.as_list()[1:])
            state_like_seqs_sliced[example_name] = sliced_seq
        for example_name, seq in action_like_seqs.items():
            sliced_seq = seq[action_like_t_slice]
            sliced_seq.set_shape([(sequence_length - 1)] + seq.shape.as_list()[1:])
            action_like_seqs_sliced[example_name] = sliced_seq

        return constant_data, state_like_seqs_sliced, action_like_seqs_sliced

    def make_features_description(self):
        features_description = {}
        for feature_name in self.trajectory_constant_names_and_shapes:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        for i in range(self.max_sequence_length):
            for feature_name in self.state_like_names_and_shapes:
                features_description[feature_name % i] = tf.io.FixedLenFeature([], tf.string)
        for i in range(self.max_sequence_length - 1):
            for feature_name in self.action_like_names_and_shapes:
                features_description[feature_name % i] = tf.io.FixedLenFeature([], tf.string)

        return features_description

    def split_into_sequences(self, example_dict):
        state_like_seqs = {}
        action_like_seqs = {}
        constant_data = {}

        for feature_name in self.state_like_names_and_shapes:
            state_like_seq = []
            for i in range(self.max_sequence_length):
                state_like_seq.append(example_dict[feature_name % i])
            state_like_seqs[strip_time_format(feature_name)] = tf.stack(state_like_seq, axis=0)

        for example_name in self.action_like_names_and_shapes:
            action_like_seq = []
            for i in range(self.max_sequence_length - 1):
                action_like_seq.append(example_dict[example_name % i])
            action_like_seqs[strip_time_format(example_name)] = tf.stack(action_like_seq, axis=0)

        for example_name in self.trajectory_constant_names_and_shapes:
            constant_data[example_name] = example_dict[example_name]

        return constant_data, state_like_seqs, action_like_seqs

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        # No-Op
        return dataset


def strip_time_format(feature_name):
    if feature_name.startswith('%d/'):
        return feature_name[3:]


def make_name_singular(feature_name):
    if "/" in feature_name:
        base_feature_name, postfix = feature_name.split("/")
        if not base_feature_name.endswith("_s"):
            raise ValueError("time-indexed feature name {} doesn't end with _s ".format(base_feature_name))
        base_feature_name_singular = base_feature_name[:-2]
        feature_name_singular = "{}/{}".format(base_feature_name_singular, postfix)
        next_feature_name_singular = "{}_next/{}".format(base_feature_name_singular, postfix)
    else:
        base_feature_name = feature_name
        if not base_feature_name.endswith("_s"):
            raise ValueError("time-indexed feature name {} doesn't end with _s ".format(base_feature_name))
        base_feature_name_singular = base_feature_name[:-2]
        feature_name_singular = base_feature_name_singular
        next_feature_name_singular = "{}_next".format(base_feature_name_singular)
    return feature_name_singular, next_feature_name_singular
