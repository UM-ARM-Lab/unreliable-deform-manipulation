#!/usr/bin/env python
from __future__ import division, print_function

import glob
import os
import random
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToDict
from tensorflow.contrib.training import HParams


def make_mask(T, S):
    P = T - S + 1
    mask = np.zeros((T, T))
    for i in range(S):
        mask += np.diag(np.ones(T - i), -i)
    return mask[:, :P]


class BaseStateSpaceDataset(object):
    def __init__(self, input_dir, mode='train', num_epochs=None, seed=None,
                 hparams_dict=None, hparams=None):
        """
        Args:
            input_dir: either a directory containing subdirectories train,
                val, test, etc, or a directory containing the tfrecords.
            mode: either train, val, or test
            num_epochs: if None, dataset is iterated indefinitely.
            seed: random seed for the op that samples subsequences.
            hparams_dict: a dict of `name=value` pairs, where `name` must be
                defined in `self.get_default_hparams()`.
            hparams: a string of comma separated list of `name=value` pairs,
                where `name` must be defined in `self.get_default_hparams()`.
                These values overrides any values in hparams_dict (if any).

        Note:
            self.input_dir is the directory containing the tfrecords.
        """
        self.input_dir = os.path.normpath(os.path.expanduser(input_dir))
        self.mode = mode
        self.num_epochs = num_epochs
        self.seed = seed
        self._max_sequence_length = None

        if self.mode not in ('train', 'val', 'test'):
            raise ValueError('Invalid mode %s' % self.mode)

        if not os.path.exists(self.input_dir):
            raise FileNotFoundError("input_dir %s does not exist" % self.input_dir)
        self.filenames = None
        # look for tfrecords in input_dir and input_dir/mode directories
        for input_dir in [self.input_dir, os.path.join(self.input_dir, self.mode)]:
            filenames = glob.glob(os.path.join(input_dir, '*.tfrecord*'))
            if filenames:
                self.input_dir = input_dir
                self.filenames = sorted(filenames)  # ensures order is the same across systems
                break
        if not self.filenames:
            raise FileNotFoundError('No tfrecords were found in %s.' % self.input_dir)
        self.dataset_name = os.path.basename(os.path.split(self.input_dir)[0])

        self.state_like_names_and_shapes = OrderedDict()
        self.action_like_names_and_shapes = OrderedDict()
        self.trajectory_constant_names_and_shapes = OrderedDict()

        self.hparams = self.parse_hparams(hparams_dict, hparams)
        self.start_mask = None

    def get_default_hparams_dict(self):
        """
        Returns:
            A dict with the following hyperparameters.
            sequence_length: the number of steps in the sequence, so
                state-like sequences are of length sequence_length and
                action-like sequences are of length sequence_length - 1.
            shuffle_on_val: whether to shuffle the samples regardless if mode
                is 'train' or 'val'. Shuffle never happens when mode is 'test'.
            free_space_only: when True, only sequences which contain data where the 'constraint' feature is true will be selected
        """
        hparams = dict(
            sequence_length=0,
            shuffle_on_val=False,
            free_space_only=False,
            compression_type='',
            dt=0.1,
            env_w=1.0,
            env_h=1.0,
            sdf_w=1.0,
            sdf_h=1.0,
        )
        return hparams

    def get_default_hparams(self):
        return HParams(**self.get_default_hparams_dict())

    def parse_hparams(self, hparams_dict, hparams):
        parsed_hparams = self.get_default_hparams().override_from_dict(hparams_dict or {})
        if hparams:
            if not isinstance(hparams, (list, tuple)):
                hparams = [hparams]
            for hparam in hparams:
                parsed_hparams.parse(hparam)
        return parsed_hparams

    def set_sequence_length(self, sequence_length):
        self.hparams.sequence_length = sequence_length

    def parser(self, serialized_example):
        """
        Parses a single tf.train.Example or tf.train.SequenceExample into
        configurations, actions, etc tensors.
        """
        raise NotImplementedError

    def make_dataset(self,
                     batch_size: int,
                     shuffle: bool = True,
                     p: int = None) -> tf.data.Dataset:
        """

        :param batch_size:
        :param shuffle: True or False
        :param p: number of parallel calls
        :return: dataset
        """
        filenames = self.filenames
        if shuffle:
            shuffle = (self.mode == 'train') or (self.mode == 'val' and self.hparams.shuffle_on_val)
            if shuffle:
                random.shuffle(filenames)

        dataset = tf.data.TFRecordDataset(filenames, buffer_size=8 * 1024 * 1024, compression_type=self.hparams.compression_type)

        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1024, count=self.num_epochs))
        else:
            dataset = dataset.repeat(self.num_epochs)

        def _parser(serialized_example):
            state_like_seqs, action_like_seqs = self.parser(serialized_example)
            return state_like_seqs, action_like_seqs

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

        def _reorganize_dict(state_like_sliced_seqs, action_like_sliced_seqs):
            input_dict = {}
            for k, v in state_like_sliced_seqs.items():
                # chop off the last time step since that's not part of the input
                input_dict[k] = v[:-1]
            output_dict = {'output_states': state_like_sliced_seqs['state']}
            input_dict.update(action_like_sliced_seqs)
            return input_dict, output_dict

        def _slice_sequences(state_like_seqs, action_like_seqs):
            return self.slice_sequences(state_like_seqs, action_like_seqs)

        dataset = dataset.map(_parser, num_parallel_calls=p)

        if self.hparams.free_space_only:
            dataset = dataset.filter(_filter_free_space_only)

        dataset = dataset.map(_slice_sequences, num_parallel_calls=p)
        dataset = dataset.map(_reorganize_dict, num_parallel_calls=p)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(batch_size)
        return dataset

    def make_batch(self, batch_size, shuffle=True):
        dataset = self.make_dataset(batch_size, shuffle)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

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

    def slice_sequences(self, state_like_seqs, action_like_seqs):
        """
        Slices sequences of length `example_sequence_length` into subsequences
        of length `sequence_length`. The dicts of sequences are updated
        in-place and the same dicts are returned.
        """
        sequence_length = self.hparams.sequence_length

        def choose_random_valid_start_index(constraints_seq):
            valid_start_onehot = constraints_seq.squeeze() @ self.start_mask
            valid_start_indeces = np.argwhere(valid_start_onehot == 0)
            valid_start_indeces = np.atleast_1d(valid_start_indeces.squeeze())
            choice = np.random.choice(valid_start_indeces)
            return choice

        if self.hparams.free_space_only:
            t_start = tf.py_func(choose_random_valid_start_index,
                                 [state_like_seqs['constraints']],
                                 tf.int64, name='choose_valid_start_t')
        else:
            t_start = 0

        state_like_t_slice = slice(t_start, t_start + sequence_length, 1)
        action_like_t_slice = slice(t_start, t_start + sequence_length - 1)

        state_like_sliced_seqs = OrderedDict()
        action_like_sliced_seqs = OrderedDict()
        for example_name, seq in state_like_seqs.items():
            sliced_seq = seq[state_like_t_slice]
            sliced_seq.set_shape([sequence_length] + seq.shape.as_list()[1:])
            state_like_sliced_seqs[example_name] = sliced_seq
        for example_name, seq in action_like_seqs.items():
            sliced_seq = seq[action_like_t_slice]
            sliced_seq.set_shape([(sequence_length - 1)] + seq.shape.as_list()[1:])
            action_like_sliced_seqs[example_name] = sliced_seq

        return state_like_sliced_seqs, action_like_sliced_seqs

    def num_examples_per_epoch(self):
        raise NotImplementedError


class StateSpaceDataset(BaseStateSpaceDataset):
    """
    This class supports reading tfrecords where a sequence is stored as
    multiple tf.train.Example and each of them is stored under a different
    feature name (which is indexed by the time step).
    """

    def __init__(self, *args, **kwargs):
        super(StateSpaceDataset, self).__init__(*args, **kwargs)
        self._dict_message = None

    def _infer_seq_length_and_setup(self):
        """
        Should be called after state_like_names_and_shapes and
        action_like_names_and_shapes have been finalized.
        """
        options = tf.python_io.TFRecordOptions(compression_type=self.hparams.compression_type)
        example = next(tf.python_io.tf_record_iterator(self.filenames[0], options=options))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        self._max_sequence_length = 0
        for feature_name in feature.keys():
            try:
                # plus 1 because time is 0 indexed here
                time_str = feature_name.split("/")[0]
                self._max_sequence_length = max(self._max_sequence_length, int(time_str) + 1)
            except ValueError:
                pass

        # set sequence_length to the longest possible if it is not specified
        if not self.hparams.sequence_length:
            self.hparams.sequence_length = (self._max_sequence_length - 1) + 1

        self.start_mask = make_mask(self._max_sequence_length, self.hparams.sequence_length)

    def parser(self, serialized_example):
        """
        Parses a single tf.train.Example into configurations, actions, etc tensors.
        """
        features = dict()
        for example_name, (name, shape) in self.trajectory_constant_names_and_shapes.items():
            features[name] = tf.FixedLenFeature(shape, tf.float32)
        for example_name, (name, shape) in self.trajectory_constant_names_and_shapes.items():
            features[name] = tf.FixedLenFeature(shape, tf.float32)

        for i in range(self._max_sequence_length):
            for example_name, (name, shape) in self.state_like_names_and_shapes.items():
                # FIXME: support loading of int64 features
                features[name % i] = tf.FixedLenFeature(shape, tf.float32)
        for i in range(self._max_sequence_length - 1):
            for example_name, (name, shape) in self.action_like_names_and_shapes.items():
                features[name % i] = tf.FixedLenFeature(shape, tf.float32)

        # parse all the features of all time steps together
        features = tf.parse_single_example(serialized_example, features=features)

        state_like_seqs = OrderedDict([(example_name, []) for example_name in self.state_like_names_and_shapes])
        action_like_seqs = OrderedDict([(example_name, []) for example_name in self.action_like_names_and_shapes])

        for i in range(self._max_sequence_length):
            for example_name, (name, shape) in self.state_like_names_and_shapes.items():
                state_like_seqs[example_name].append(features[name % i])
            for example_name, (name, shape) in self.trajectory_constant_names_and_shapes.items():
                if example_name not in state_like_seqs:
                    state_like_seqs[example_name] = []
                state_like_seqs[example_name].append(features[name])

        for i in range(self._max_sequence_length - 1):
            for example_name, (name, shape) in self.action_like_names_and_shapes.items():
                action_like_seqs[example_name].append(features[name % i])

        state_like_seqs, action_like_seqs = self.convert_to_sequences(state_like_seqs, action_like_seqs,
                                                                      self._max_sequence_length)
        return state_like_seqs, action_like_seqs
