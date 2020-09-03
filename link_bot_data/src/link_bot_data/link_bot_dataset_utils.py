#!/usr/bin/env python
import os
import pathlib
import time
from typing import Optional, Dict

import git
import numpy as np
import tensorflow as tf
from colorama import Fore

from link_bot_pycommon import pycommon
from link_bot_pycommon.filesystem_utils import mkdir_and_ask
from moonshine.moonshine_utils import remove_batch, add_batch

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
        for _key, _serialized_tensor in serialized_dict.items():
            _deserialized_tensor = tf.io.parse_tensor(_serialized_tensor, tf.float32)
            _deserialized_tensor = tf.ensure_shape(_deserialized_tensor, inferred_shapes[_key])
            deserialized_dict[_key] = _deserialized_tensor
        return deserialized_dict

    deserialized_dataset = parsed_dataset.map(_deserialize, num_parallel_calls=n_parallel_calls)
    return deserialized_dataset


def dict_of_float_tensors_to_bytes_feature(d):
    return {k: float_tensor_to_bytes_feature(v) for k, v in d.items()}


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


def batch_tf_dataset(dataset: tf.data.Dataset, batch_size: int, drop_remainder: bool = True):
    def _add_batch(example: Dict):
        example['batch_size'] = tf.cast(batch_size, tf.int64)
        return example

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.map(_add_batch)
    return dataset


def filter_and_cache(dataset, filter_func):
    dataset = dataset.filter(filter_func)
    dataset = dataset.cache(cachename())
    return dataset


def balance(dataset, labeling_params: Dict, cache_negative: bool = True):
    def _label_is(label_is):
        def __filter(transition):
            result = tf.squeeze(tf.equal(transition['label'], label_is))
            return result

        return __filter

    positive_examples = dataset.filter(_label_is(1))
    negative_examples = dataset.filter(_label_is(0))
    if cache_negative:
        negative_examples = negative_examples.cache(cachename())

    # Combine and flatten
    # zip will balance by dropping examples from whichever is larger
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
        tmpname = cache_root / f"{mode}_{pycommon.rand_str()}"
    else:
        tmpname = cache_root / f"{pycommon.rand_str()}"
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


PREDICTED_PREFIX = 'predicted/'


def remove_predicted(k: str):
    if k.startswith(PREDICTED_PREFIX):
        return k[len(PREDICTED_PREFIX):]
    else:
        return k


def remove_predicted_from_dict(d: Dict):
    return {remove_predicted(k): v for k, v in d.items()}


def add_predicted(feature_name):
    return PREDICTED_PREFIX + feature_name


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


def is_reconverging(labels, label_threshold=0.5):
    """
    :param labels: a [B, H] matrix of 1/0 or true/false
    :param label_threshold: numbers above this threshold are considered true
    :return: a [B] binary vector with [i] true for if labels[i] is reconverging
    """
    float_labels = tf.cast(labels, tf.float32)
    int_labels = tf.cast(labels, tf.int64)
    starts_with_1 = float_labels[:, 0] > label_threshold
    ends_with_1 = float_labels[:, -1] > label_threshold
    num_ones = tf.reduce_sum(int_labels, axis=1)
    index_of_last_1 = float_labels.shape[1] - tf.argmax(tf.reverse(float_labels, axis=[1]), axis=1) - 1
    reconverging = (index_of_last_1 >= num_ones)
    reconverging_and_start_end_1 = tf.stack([reconverging, starts_with_1, ends_with_1], axis=1)
    return tf.reduce_all(reconverging_and_start_end_1, axis=1)


def num_reconverging(labels):
    """
    :param labels: [B, H] matrix
    :return:
    """

    return tf.math.reduce_sum(tf.cast(is_reconverging(labels), dtype=tf.int32))


def num_reconverging_subsequences(labels):
    """
    :param labels: [B, H] matrix
    :return:
    """
    n = 0
    for start_idx in range(labels.shape[1]):
        for end_idx in range(start_idx + 2, labels.shape[1] + 1):
            n_i = num_reconverging(labels[:, start_idx:end_idx])
            n += n_i
    return n


def filter_only_reconverging(example):
    is_close = example['is_close']
    return remove_batch(is_reconverging(add_batch(is_close)))


def filter_no_reconverging(example):
    is_close = example['is_close']
    return tf.logical_not(remove_batch(is_reconverging(add_batch(is_close))))


def label_is(label_is, key='is_close'):
    def __filter(example):
        result = tf.squeeze(tf.equal(example[key][1], label_is))
        return result

    return __filter


def flatten_concat_pairs(ex_pos, ex_neg):
    flat_pair = tf.data.Dataset.from_tensors(ex_pos).concatenate(tf.data.Dataset.from_tensors(ex_neg))
    return flat_pair


def balance(dataset):
    # FIXME: redo this when I redo my dataset code
    positive_examples = dataset.filter(label_is(1))
    negative_examples = dataset.filter(label_is(0))
    # negative_examples = negative_examples.repeat()
    print("UP-SAMPLING POSITIVE EXAMPLES!!!")
    positive_examples = positive_examples.repeat()
    balanced_dataset = tf.data.Dataset.zip((positive_examples, negative_examples))
    balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)

    return balanced_dataset
