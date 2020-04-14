#!/usr/bin/env python
import os
import pathlib
from typing import Optional, Dict

import git
import numpy as np
import tensorflow as tf
from colorama import Fore

from link_bot_pycommon import link_bot_pycommon

NULL_PAD_VALUE = -10000


def parse_and_deserialize(dataset, feature_description, n_parallel_calls=None):
    parsed_dataset = parse_dataset(dataset, feature_description, n_parallel_calls=n_parallel_calls)
    deserialized_dataset = deserialize(parsed_dataset, n_parallel_calls=n_parallel_calls)
    return deserialized_dataset


def parse_dataset(dataset, feature_description, n_parallel_calls=None):
    def _parse(example_proto):
        deserialized_dict = tf.io.parse_single_example(example_proto, feature_description)
        return deserialized_dict

    # the elements of parsed dataset are dictionaries with the serialized tensors as strings
    parsed_dataset = dataset.map(_parse, num_parallel_calls=n_parallel_calls)
    return parsed_dataset


def deserialize(parsed_dataset, n_parallel_calls=None):
    # get shapes of everything
    element = next(iter(parsed_dataset))
    inferred_shapes = {}
    for key, serialized_tensor in element.items():
        deserialized_tensor = tf.io.parse_tensor(serialized_tensor, tf.float32)
        inferred_shapes[key] = deserialized_tensor.shape

    def _deserialize(serialized_dict):
        deserialized_dict = {}
        for key, serialized_tensor in serialized_dict.items():
            deserialized_tensor = tf.io.parse_tensor(serialized_tensor, tf.float32)
            deserialized_tensor = tf.ensure_shape(deserialized_tensor, inferred_shapes[key])
            deserialized_dict[key] = deserialized_tensor
        return deserialized_dict

    deserialized_dataset = parsed_dataset.map(_deserialize, num_parallel_calls=n_parallel_calls)
    return deserialized_dataset


def float_tensor_to_bytes_feature(value):
    return bytes_feature(tf.io.serialize_tensor(tf.convert_to_tensor(value, dtype=tf.float32)).numpy())


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int_feature(values):
    """Returns a int64 from 1-dimensional numpy array"""
    assert values.ndim == 1
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    """Returns a float_list from 1-dimensional numpy array"""
    assert values.ndim == 1
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def flatten_concat_pairs(ex_pos, ex_neg):
    flat_pair = tf.data.Dataset.from_tensors(ex_pos).concatenate(tf.data.Dataset.from_tensors(ex_neg))
    return flat_pair


def has_already_diverged(transition: Dict, labeling_params):
    state_key = labeling_params['state_key']

    planned_state_all = transition[add_all_and_planned(state_key)]
    actual_state_all = transition[add_all(state_key)]

    pre_threshold = labeling_params['pre_close_threshold']
    pre_close = tf.norm(planned_state_all - actual_state_all, axis=1)[:-1] < pre_threshold
    next_is_null = tf.reduce_all(tf.equal(planned_state_all, NULL_PAD_VALUE), axis=1)[1:]
    already_diverged = tf.reduce_any(tf.logical_not(tf.logical_or(pre_close, next_is_null)))

    return already_diverged


def balance(dataset, labeling_params: Dict):
    def _label_is(label_is):
        def __filter(transition):
            label_key = labeling_params['label_key']
            result = tf.squeeze(tf.equal(transition[label_key], label_is))
            return result

        return __filter

    def _filter_out_already_diverged(transition):
        return tf.logical_not(has_already_diverged(transition, labeling_params))

    positive_examples = dataset.filter(_label_is(1))
    negative_examples = dataset.filter(_label_is(0))
    negative_examples = negative_examples.cache(cachename())

    # now split filter out examples where the prediction diverged previously in the trajectory
    if labeling_params['discard_pre_far']:
        negative_examples = negative_examples.filter(_filter_out_already_diverged)
        # cache again for efficiency, since filter is slow when most elements are filtered out (which they are here)
        negative_examples = negative_examples.cache(cachename())

    # TODO: check which dataset is smaller, and repeat that one
    negative_examples = negative_examples.repeat()

    # Combine and flatten
    balanced_dataset = tf.data.Dataset.zip((positive_examples, negative_examples))
    balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)

    return balanced_dataset


def cachename(mode: Optional[str] = None):
    if mode is not None:
        tmpname = "/data/tf_{}_{}".format(mode, link_bot_pycommon.rand_str())
    else:
        tmpname = "/tmp/tf_{}".format(link_bot_pycommon.rand_str())
    return tmpname


def data_directory(outdir: pathlib.Path, *names):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    format_string = "{}_{}_" + "{}_" * (len(names) - 1) + "{}"
    full_output_directory = pathlib.Path(format_string.format(outdir, sha, *names))
    if outdir:
        if full_output_directory.is_file():
            print(Fore.RED + "argument outdir is an existing file, aborting." + Fore.RESET)
            return
        elif not full_output_directory.is_dir():
            os.mkdir(full_output_directory)
    return full_output_directory


def add_next(feature_name):
    return feature_name + '_next'


def add_all(feature_name):
    return feature_name + '_all'


def add_planned(feature_name):
    return "planned_state/" + feature_name


def add_next_and_planned(feature_name):
    return add_next(add_planned(feature_name))


def add_all_and_planned(feature_name):
    return add_all(add_planned(feature_name))


def null_future_states(sequence, end_idx):
    if isinstance(sequence, tf.Tensor):
        sequence = sequence.numpy()
    new_sequence = sequence.copy()
    if end_idx + 1 < len(sequence):
        new_sequence[end_idx + 1:] = NULL_PAD_VALUE
    return new_sequence


def null_previous_states(example, max_sequence_length):
    padded_example = {}
    for k, v in example.items():
        v = v.numpy()
        new_shape = [v.shape[0], max_sequence_length - v.shape[1]]
        if len(v.shape) > 2:
            new_shape.extend(v.shape[2:])
        nulls = np.ones(new_shape) * NULL_PAD_VALUE
        padded_example[k] = np.concatenate((nulls, v), axis=1)

    return padded_example


def split_into_sequences(state_feature_names, action_feature_names, constant_feature_names, max_sequence_length, example_dict):
    state_like_seqs = {}
    action_like_seqs = {}
    constant_data = {}

    for feature_name in state_feature_names:
        feature_name = "%d/" + feature_name
        state_like_seq = []
        for i in range(max_sequence_length):
            state_like_seq.append(example_dict[feature_name % i])
        state_like_seqs[strip_time_format(feature_name)] = tf.stack(state_like_seq, axis=0)

    for example_name in action_feature_names:
        example_name = "%d/" + example_name
        action_like_seq = []
        for i in range(max_sequence_length - 1):
            action_like_seq.append(example_dict[example_name % i])
        action_like_seqs[strip_time_format(example_name)] = tf.stack(action_like_seq, axis=0)

    for example_name in constant_feature_names:
        constant_data[example_name] = example_dict[example_name]

    return constant_data, state_like_seqs, action_like_seqs


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


def strip_time_format(feature_name):
    if feature_name.startswith('%d/'):
        return feature_name[3:]
