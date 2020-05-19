#!/usr/bin/env python
import os
import pathlib
import time
from typing import Optional, Dict

import git
import numpy as np
import tensorflow as tf
from colorama import Fore

from link_bot_pycommon import link_bot_pycommon
from link_bot_pycommon.filesystem_utils import mkdir_and_ask

NULL_PAD_VALUE = -10000


def state_dict_is_null(state: Dict):
    for v in state.values():
        if np.any(v == NULL_PAD_VALUE):
            return True
    return False


def state_dict_is_null_tf(state: Dict):
    for v in state.values():
        if tf.reduce_any(tf.equal(v, NULL_PAD_VALUE)):
            return True
    return False


def total_state_dim(state: Dict):
    """
    :param state: assumed to be [batch, state_dim]
    :return:
    """
    state_dim = 0
    for v in state.values():
        state_dim += int(v.shape[1] / 2)
    return state_dim


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
    start_t = tf.cast(transition['start_t'], tf.int64)
    end_t = tf.cast(transition['end_t'], tf.int64)

    planned_state_all = transition[add_all_and_planned(state_key)]
    actual_state_all = transition[add_all(state_key)]

    pre_threshold = labeling_params['pre_close_threshold']
    pre_far = tf.norm(planned_state_all - actual_state_all, axis=1) > pre_threshold
    # this will not include $\hat{s}^{t+1}$, aka planned_state/next... which is what we want
    # since here we only require that all states BEFORE the final state are accurate.
    # if it is inaccurate/diverges at the last state pair that's fine, that's what a 0 label is.
    already_diverged = tf.reduce_any(pre_far[start_t:end_t])

    return already_diverged


def balance(dataset, labeling_params: Dict):
    def _label_is(label_is):
        def __filter(transition):
            label_key = labeling_params['label_key']
            result = tf.squeeze(tf.equal(transition[label_key], label_is))
            return result

        return __filter

    positive_examples = dataset.filter(_label_is(1))
    negative_examples = dataset.filter(_label_is(0))
    negative_examples = negative_examples.cache(cachename())

    # TODO: check which dataset is smaller, and repeat that one
    negative_examples = negative_examples.repeat()

    # Combine and flatten
    balanced_dataset = tf.data.Dataset.zip((positive_examples, negative_examples))
    balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)

    return balanced_dataset


def cachename(mode: Optional[str] = None):
    if 'TF_CACHE_ROOT' in os.environ:
        cache_root = pathlib.Path(os.environ['TF_CACHE_ROOT'])
        cache_root.mkdir(exist_ok=True, parents=True)
    else:
        cache_root = pathlib.Path('/tmp')
    if mode is not None:
        tmpname = cache_root / f"{mode}_{link_bot_pycommon.rand_str()}"
    else:
        tmpname = cache_root / f"{link_bot_pycommon.rand_str()}"
    return str(tmpname)


def data_directory(outdir: pathlib.Path, *names):
    now = str(int(time.time()))
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    format_string = "{}_{}_{}" + len(names) * '_{}'
    full_output_directory = pathlib.Path(format_string.format(outdir, now, sha, *names))
    if outdir:
        if full_output_directory.is_file():
            print(Fore.RED + "argument outdir is an existing file, aborting." + Fore.RESET)
            return
        elif not full_output_directory.is_dir():
            mkdir_and_ask(full_output_directory, parents=True)
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


def null_pad(sequence, start=None, end=None):
    if isinstance(sequence, tf.Tensor):
        new_sequence = sequence.numpy().copy()
    else:
        new_sequence = sequence.copy()
    if start is not None and start > 0:
        new_sequence[:start] = NULL_PAD_VALUE
    if end is not None and end + 1 < len(sequence):
        new_sequence[end + 1:] = NULL_PAD_VALUE
    return new_sequence


def null_future_states(sequence, end_idx):
    if isinstance(sequence, tf.Tensor):
        new_sequence = sequence.numpy().copy()
    else:
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


def null_diverged(true_sequences, pred_sequences, start_t: int, labeling_params: Dict):
    threshold = labeling_params['threshold']
    state_key = labeling_params['state_key']

    pred_sequence_for_state_key = pred_sequences[state_key]
    sequence_for_state_key = true_sequences[state_key][:, start_t:]
    # stop the trajectory at the first divergence. i.e the first time d(s^{t+1},\hat{s}^{t+1}) > delta
    model_error = tf.linalg.norm(sequence_for_state_key - pred_sequence_for_state_key, axis=2)
    close = tf.cast(model_error < threshold, dtype=tf.int32)
    expected_cumsum = tf.math.cumsum(tf.ones_like(close, dtype=tf.int32), axis=1)
    close_cumsum = tf.math.cumsum(close, axis=1)
    not_diverged_yet_mask = tf.expand_dims(tf.cast(tf.equal(expected_cumsum - close_cumsum, 0), tf.float32), axis=2)
    # shift all columns to the right, add a column of ones to the front. This allows us to include the first diverged state
    # which we need for label=0 examples
    not_diverged_yet_mask = tf.concat((tf.ones([not_diverged_yet_mask.shape[0], 1, 1]), not_diverged_yet_mask[:, :-1]), axis=1)
    has_diverged_mask = 1.0 - not_diverged_yet_mask

    last_valid_ts = tf.squeeze(tf.reduce_sum(not_diverged_yet_mask, axis=1), axis=1) - 1 + start_t

    # using :first_diverged_t will now get what we want, where the final transition is either the end of the sequence of the first
    # diverged transition
    all_pred_sequence_masked = {}
    for key, v in pred_sequences.items():
        all_pred_sequence_masked[key] = (not_diverged_yet_mask * v) + (has_diverged_mask * NULL_PAD_VALUE)

    return all_pred_sequence_masked, last_valid_ts


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


def slice_sequences(constant_data, state_like_seqs, action_like_seqs, desired_sequence_length: int, step_size: int):
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
    for t_start in range(0, sequence_length, step_size):
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


def is_reconverging(labels):
    num_ones = tf.reduce_sum(labels)
    index_of_last_1 = tf.reduce_max(tf.where(labels))
    reconverging = (index_of_last_1 >= num_ones)
    return reconverging
