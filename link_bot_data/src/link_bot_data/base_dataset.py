#!/usr/bin/env python
from __future__ import division, print_function

import json
import pathlib
from typing import List, Optional

import tensorflow as tf
from colorama import Fore

from link_bot_data.link_bot_dataset_utils import parse_and_deserialize, parse_dataset

DEFAULT_VAL_SPLIT = 0.125
DEFAULT_TEST_SPLIT = 0.125


def slice_sequences(constant_data, state_like_seqs, action_like_seqs, desired_sequence_length: int):
    sequence_length = int(next(iter((state_like_seqs.values()))).shape[0])

    state_like_seqs_sliced = {}
    action_like_seqs_sliced = {}
    constant_data_sliced = {}

    # pre-create the lists
    for example_name, seq in state_like_seqs.items():
        state_like_seqs_sliced[example_name] = []
    for example_name, seq in action_like_seqs.items():
        action_like_seqs_sliced[example_name] = []
    for example_name, constant_datum in constant_data.items():
        constant_data_sliced[example_name] = []

    # add elements to the lists
    for t_start in range(0, sequence_length, desired_sequence_length):
        if t_start + desired_sequence_length > sequence_length:
            break
        state_like_t_slice = slice(t_start, t_start + desired_sequence_length)
        action_like_t_slice = slice(t_start, t_start + desired_sequence_length - 1)

        for example_name, seq in state_like_seqs.items():
            sliced_seq = seq[state_like_t_slice]
            state_like_seqs_sliced[example_name].append(sliced_seq)

        for example_name, seq in action_like_seqs.items():
            sliced_seq = seq[action_like_t_slice]
            action_like_seqs_sliced[example_name].append(sliced_seq)

        for example_name, constant_datum in constant_data.items():
            constant_data_sliced[example_name].append(constant_datum)

    # we need to return a 3-tuple dictionary where every key has the same first dimension
    return tf.data.Dataset.from_tensor_slices((constant_data_sliced, state_like_seqs_sliced, action_like_seqs_sliced))


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

        self.max_sequence_length = self.hparams['sequence_length']

        # state and action features are assumed to be time indexed, i.e. "%d/my_state"
        # state and action are handled differently because there should always be one less action
        # in the sequence than there are states
        self.state_feature_names = {}
        self.action_feature_names = {}
        self.constant_feature_names = {}

    def get_datasets(self,
                     mode: str,
                     sequence_length: Optional[int] = None,
                     n_parallel_calls: int = tf.data.experimental.AUTOTUNE,
                     do_not_process: bool = False,
                     take: Optional[int] = None,
                     ) -> tf.data.Dataset:
        if mode == 'all':
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
        else:
            all_filenames = []
            for dataset_dir in self.dataset_dirs:
                all_filenames.extend(str(filename) for filename in (dataset_dir / mode).glob("*.tfrecords"))

        desired_sequence_length = sequence_length if sequence_length is not None else self.max_sequence_length
        return self.get_datasets_from_records(all_filenames,
                                              desired_sequence_length=desired_sequence_length,
                                              n_parallel_calls=n_parallel_calls,
                                              do_not_process=do_not_process,
                                              take=take)

    def get_datasets_from_records(self,
                                  records: List[str],
                                  desired_sequence_length: Optional[int] = None,
                                  n_parallel_calls: Optional[int] = None,
                                  do_not_process: Optional[bool] = False,
                                  take: Optional[int] = None,
                                  ) -> tf.data.Dataset:
        dataset = tf.data.TFRecordDataset(records, buffer_size=1 * 1024 * 1024, compression_type='ZLIB')

        # Given the member lists of states, actions, and constants set in the constructor, create a dict for parsing a feature
        features_description = self.make_features_description()
        dataset = parse_and_deserialize(dataset, feature_description=features_description, n_parallel_calls=n_parallel_calls)

        if take is not None:
            dataset = dataset.take(take)

        if not do_not_process:
            dataset = dataset.map(self.split_into_sequences, num_parallel_calls=n_parallel_calls)

            def _slice_sequences(constant_data, state_like_seqs, action_like_seqs):
                return slice_sequences(constant_data, state_like_seqs, action_like_seqs, desired_sequence_length)

            dataset = dataset.flat_map(_slice_sequences)

            dataset = self.post_process(dataset, n_parallel_calls)

        return dataset

    def old_make_features_description(self):
        hacky_lookup = {
            'image': [50, 50, 23],
            'action': 2,
            'actual_local_env/env': [50, 50],
            'actual_local_env/extent': 4,
            'actual_local_env/origin': 2,
            'actual_local_env_next/env': [50, 50],
            'actual_local_env_next/extent': 4,
            'actual_local_env_next/origin': 2,
            'full_env/env': [200, 200],
            'label': 1,
            'local_env_cols': 1,
            'local_env_rows': 1,
            'planned_local_env/env': [50, 50],
            'planned_local_env/extent': 4,
            'planned_local_env/origin': 2,
            'planned_local_env_next/env': [50, 50],
            'planned_local_env_next/extent': 4,
            'planned_local_env_next/origin': 2,
            'planned_state': 22,
            'planned_state_next': 22,
            'post_dist': 1,
            'pre_close': 1,
            'pre_dist': 1,
            'resolution': 1,
            'resolution_next': 1,
            'state': 22,
            'full_env/extent': 4,
            'full_env/origin': 2,
            '%d/res': 1,
            '%d/actual_local_env/env': 2500,
            '%d/actual_local_env/extent': 4,
            '%d/actual_local_env/origin': 2,
            '%d/planned_local_env/env': 2500,
            '%d/planned_local_env/extent': 4,
            '%d/planned_local_env/origin': 2,
            '%d/planned_state': 22,
            '%d/state': 22,
            '%d/traj_idx': 1,
            '%d/time_idx': 1,
            '%d/time_idx ': 1,
            '%d/action': 2,
            '%d/1/force': 2,
            '%d/1/velocity': 2,
            '%d/1/post_action_velocity': 2,
            '%d/endeffector_pos': 2,
            '%d/constraint': 1,
            '%d/rope_configuration': 22,
        }

        features_description = {}
        for feature_name in self.constant_feature_names:
            shape = hacky_lookup[feature_name]
            features_description[feature_name] = tf.io.FixedLenFeature(shape, tf.float32)

        for i in range(self.max_sequence_length):
            for feature_name in self.state_feature_names:
                feature_name = "%d/" + feature_name
                shape = hacky_lookup[feature_name]
                features_description[feature_name % i] = tf.io.FixedLenFeature(shape, tf.float32)
        for i in range(self.max_sequence_length - 1):
            for feature_name in self.action_feature_names:
                feature_name = "%d/" + feature_name
                shape = hacky_lookup[feature_name]
                features_description[feature_name % i] = tf.io.FixedLenFeature(shape, tf.float32)

        return features_description

    def make_features_description(self):
        features_description = {}
        for feature_name in self.constant_feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        for i in range(self.max_sequence_length):
            for feature_name in self.state_feature_names:
                feature_name = "%d/" + feature_name
                features_description[feature_name % i] = tf.io.FixedLenFeature([], tf.string)
        for i in range(self.max_sequence_length - 1):
            for feature_name in self.action_feature_names:
                feature_name = "%d/" + feature_name
                features_description[feature_name % i] = tf.io.FixedLenFeature([], tf.string)

        return features_description

    def split_into_sequences(self, example_dict):
        state_like_seqs = {}
        action_like_seqs = {}
        constant_data = {}

        for feature_name in self.state_feature_names:
            feature_name = "%d/" + feature_name
            state_like_seq = []
            for i in range(self.max_sequence_length):
                state_like_seq.append(example_dict[feature_name % i])
            state_like_seqs[strip_time_format(feature_name)] = tf.stack(state_like_seq, axis=0)

        for example_name in self.action_feature_names:
            example_name = "%d/" + example_name
            action_like_seq = []
            for i in range(self.max_sequence_length - 1):
                action_like_seq.append(example_dict[example_name % i])
            action_like_seqs[strip_time_format(example_name)] = tf.stack(action_like_seq, axis=0)

        for example_name in self.constant_feature_names:
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
