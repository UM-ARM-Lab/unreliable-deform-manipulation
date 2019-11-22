#!/usr/bin/env python
from __future__ import division, print_function

import json
import pathlib
from typing import List, Optional

import numpy as np
import tensorflow as tf
from colorama import Fore
from google.protobuf.json_format import MessageToDict

from link_bot_data.link_bot_dataset_utils import balance_xy_dataset, balance_x_dataset


def make_mask(T, S):
    P = T - S + 1
    mask = np.zeros((T, T))
    for i in range(S):
        mask += np.diag(np.ones(T - i), -i)
    return mask[:, :P]


def get_max_sequence_length(records, compression_type):
    options = tf.io.TFRecordOptions(compression_type=compression_type)
    example = next(tf.io.tf_record_iterator(records[0], options=options))
    dict_message = MessageToDict(tf.train.Example.FromString(example))
    feature = dict_message['features']['feature']
    max_sequence_length = 0
    for feature_name in feature.keys():
        try:
            # plus 1 because time is 0 indexed here
            time_str = feature_name.split("/")[0]
            max_sequence_length = max(max_sequence_length, int(time_str) + 1)
        except ValueError:
            pass
    return max_sequence_length


class BaseStateSpaceDataset:
    def __init__(self, dataset_dir: pathlib.Path):
        self.dataset_dir = dataset_dir
        dataset_hparams_filename = dataset_dir / 'hparams.json'
        self.hparams = json.load(open(str(dataset_hparams_filename), 'r'))

        self.max_sequence_length = 0

        self.state_like_names_and_shapes = {}
        self.action_like_names_and_shapes = {}
        self.trajectory_constant_names_and_shapes = {}
        self.start_mask = None

    def parser(self, serialized_example):
        """
        Parses a single tf.train.Example or tf.train.SequenceExample into
        configurations, actions, etc tensors.
        """
        raise NotImplementedError

    def get_dataset(self,
                    mode: str,
                    batch_size: int,
                    seed: int,
                    shuffle: bool = True,
                    sequence_length: Optional[int] = None,
                    n_parallel_calls: int = 8,
                    balance_key: Optional[str] = None) -> tf.data.Dataset:
        records = [str(filename) for filename in (self.dataset_dir / mode).glob("*.tfrecords")]
        return self.get_dataset_from_records(records,
                                             batch_size=batch_size,
                                             seed=seed,
                                             shuffle=shuffle,
                                             sequence_length=sequence_length,
                                             balance_key=balance_key,
                                             n_parallel_calls=n_parallel_calls)

    def get_dataset_all_modes(self,
                              batch_size: Optional[int],
                              shuffle: bool = True,
                              seed: int = 0,
                              balance_key: Optional[str] = None):
        train_filenames = [str(filename) for filename in self.dataset_dir.glob("{}/*.tfrecords".format('train'))]
        test_filenames = [str(filename) for filename in self.dataset_dir.glob("{}/*.tfrecords".format('test'))]
        val_filenames = [str(filename) for filename in self.dataset_dir.glob("{}/*.tfrecords".format('val'))]

        all_filenames = train_filenames
        all_filenames.extend(test_filenames)
        all_filenames.extend(val_filenames)

        return self.get_dataset_from_records(records=all_filenames,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             seed=seed,
                                             balance_key=balance_key)

    def get_dataset_from_records(self,
                                 records: List[str],
                                 batch_size: int,
                                 seed: int,
                                 shuffle: bool = True,
                                 sequence_length: Optional[int] = None,
                                 balance_key: str = None,
                                 n_parallel_calls: int = None,
                                 ) -> tf.data.Dataset:

        self.max_sequence_length = get_max_sequence_length(records, self.hparams['compression_type'])

        if sequence_length is None:
            # set sequence_length to the longest possible if it is not specified
            sequence_length = self.max_sequence_length
            msg = "sequence length not specified, assuming max sequence length: {}".format(self.max_sequence_length)
            print(Fore.YELLOW + msg + Fore.RESET)

        self.start_mask = make_mask(self.max_sequence_length, sequence_length)

        dataset = tf.data.TFRecordDataset(records, buffer_size=8 * 1024 * 1024,
                                          compression_type=self.hparams['compression_type'])

        if shuffle:
            # TODO: tune buffer size for performance
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)

        def has_valid_index(constraints_seq):
            valid_start_onehot = constraints_seq.squeeze() @ self.start_mask
            valid_start_indeces = np.argwhere(valid_start_onehot == 0).squeeze()
            # Handle the case where there is no such sequence
            return valid_start_indeces.size > 0

        def _filter_free_space_only(state_like_seqs, action_like_seqs):
            del action_like_seqs
            is_valid = tf.py_func(has_valid_index,
                                  [state_like_seqs['constraints']],
                                  tf.bool, name='has_valid_index')
            return is_valid

        def _slice_sequences(constant_data, state_like_seqs, action_like_seqs):
            return self.slice_sequences(constant_data, state_like_seqs, action_like_seqs, sequence_length=sequence_length)

        dataset = dataset.map(self.parser, num_parallel_calls=n_parallel_calls)

        if self.hparams['filter_free_space_only']:
            dataset = dataset.filter(_filter_free_space_only)

        dataset = dataset.map(_slice_sequences, num_parallel_calls=n_parallel_calls)

        if balance_key is not None:
            dataset = balance_x_dataset(dataset, balance_key)

        dataset_size = 0
        for _ in dataset:
            dataset_size += batch_size

        dataset = self.post_process(dataset, n_parallel_calls)

        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(batch_size)

        # sanity check that the dataset isn't empty, which can happen when debugging if batch size is bigger than dataset size
        empty = True
        for _ in dataset:
            empty = False
            break
        if empty:
            raise RuntimeError("Dataset is empty! batch size: {}, dataset size: {}".format(batch_size, dataset_size))

        return dataset

    def convert_to_sequences(self, state_like_seqs, action_like_seqs, example_sequence_length):
        # Convert anything which is a list of tensors along time dimension to one tensor where the first dimension is time
        state_like_tensors = {}
        action_like_tensors = {}
        for example_name, seq in state_like_seqs.items():
            seq = tf.convert_to_tensor(seq)
            seq.set_shape([example_sequence_length] + seq.shape.as_list()[1:])
            state_like_tensors[example_name] = seq
        for example_name, seq in action_like_seqs.items():
            seq = tf.convert_to_tensor(seq)
            seq.set_shape([example_sequence_length - 1] + seq.shape.as_list()[1:])
            action_like_tensors[example_name] = seq

        return state_like_tensors, action_like_tensors

    def slice_sequences(self, constant_data, state_like_seqs, action_like_seqs, sequence_length: int):
        def choose_random_valid_start_index(constraints_seq):
            valid_start_onehot = constraints_seq.squeeze() @ self.start_mask
            valid_start_indeces = np.argwhere(valid_start_onehot == 0)
            valid_start_indeces = np.atleast_1d(valid_start_indeces.squeeze())
            choice = np.random.choice(valid_start_indeces)
            return choice

        if self.hparams['filter_free_space_only']:
            t_start = tf.py_func(choose_random_valid_start_index,
                                 [state_like_seqs['constraints']],
                                 tf.int64, name='choose_valid_start_t')
        else:
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

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        # No-Op
        return dataset


class StateSpaceDataset(BaseStateSpaceDataset):

    def __init__(self,
                 dataset_dir: pathlib.Path):
        super(StateSpaceDataset, self).__init__(dataset_dir)

    def parser(self, serialized_example):
        """
        Parses a single tf.train.Example into configurations, actions, etc tensors.
        """
        features = dict()
        for example_name, (name, shape) in self.trajectory_constant_names_and_shapes.items():
            features[name] = tf.io.FixedLenFeature(shape, tf.float32)

        for i in range(self.max_sequence_length):
            for example_name, (name, shape) in self.state_like_names_and_shapes.items():
                # FIXME: support loading of int64 features
                features[name % i] = tf.io.FixedLenFeature(shape, tf.float32)
        for i in range(self.max_sequence_length - 1):
            for example_name, (name, shape) in self.action_like_names_and_shapes.items():
                features[name % i] = tf.io.FixedLenFeature(shape, tf.float32)

        # parse all the features of all time steps together
        features = tf.io.parse_single_example(serialized_example, features=features)

        state_like_seqs = dict([(example_name, []) for example_name in self.state_like_names_and_shapes])
        action_like_seqs = dict([(example_name, []) for example_name in self.action_like_names_and_shapes])
        constant_data = {}

        for i in range(self.max_sequence_length):
            for example_name, (name, shape) in self.state_like_names_and_shapes.items():
                state_like_seqs[example_name].append(features[name % i])

        for i in range(self.max_sequence_length - 1):
            for example_name, (name, shape) in self.action_like_names_and_shapes.items():
                action_like_seqs[example_name].append(features[name % i])

        for example_name, (name, shape) in self.trajectory_constant_names_and_shapes.items():
            constant_data[example_name] = features[name]

        state_like_seqs, action_like_seqs = self.convert_to_sequences(state_like_seqs, action_like_seqs,
                                                                      self.max_sequence_length)
        return constant_data, state_like_seqs, action_like_seqs
