#!/usr/bin/env python
import os
import pathlib
import time
from typing import Optional, Dict

import git
import numpy as np
import tensorflow as tf
from colorama import Fore

from arc_utilities.filesystem_utils import mkdir_and_ask
from link_bot_pycommon import pycommon
from moonshine.moonshine_utils import remove_batch, add_batch

NULL_PAD_VALUE = -10000

STRING_KEYS = [
    'tfrecord_path'
]


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


def deserialize(parsed_dataset: tf.data.Dataset, n_parallel_calls=None):
    # get shapes of everything
    inferred_shapes = infer_shapes(parsed_dataset)

    def _deserialize(serialized_dict):
        deserialized_dict = {}
        for _key, _serialized_tensor in serialized_dict.items():
            if _key in STRING_KEYS:
                _deserialized_tensor = tf.io.parse_tensor(_serialized_tensor, tf.string)
            else:
                _deserialized_tensor = tf.io.parse_tensor(_serialized_tensor, tf.float32)
                _deserialized_tensor = tf.ensure_shape(_deserialized_tensor, inferred_shapes[_key])
            deserialized_dict[_key] = _deserialized_tensor
        return deserialized_dict

    deserialized_dataset = parsed_dataset.map(_deserialize, num_parallel_calls=n_parallel_calls)
    return deserialized_dataset


def infer_shapes(parsed_dataset: tf.data.Dataset):
    element = next(iter(parsed_dataset))
    inferred_shapes = {}
    for key, serialized_tensor in element.items():
        if key in STRING_KEYS:
            inferred_shapes[key] = ()
        else:
            deserialized_tensor = tf.io.parse_tensor(serialized_tensor, tf.float32)
            inferred_shapes[key] = deserialized_tensor.shape
    return inferred_shapes


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


def batch_tf_dataset(dataset: tf.data.Dataset, batch_size: int, drop_remainder: bool = True):
    def _add_batch(example: Dict):
        for v in example.values():
            v.is_batched = True
        example['batch_size'] = tf.cast(batch_size, tf.int64)
        return example

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.map(_add_batch)
    return dataset


def filter_and_cache(dataset, filter_func):
    dataset = dataset.filter(filter_func)
    dataset = dataset.cache(cachename())
    return dataset


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
POSITIVE_PREFIX = 'positive/'
NEXT_PREFIX = 'next/'


def remove_predicted(k: str):
    if k.startswith(PREDICTED_PREFIX):
        return k[len(PREDICTED_PREFIX):]
    else:
        return k


def remove_predicted_from_dict(d: Dict):
    return {remove_predicted(k): v for k, v in d.items()}


def add_positive(feature_name):
    return POSITIVE_PREFIX + feature_name


def add_next(feature_name):
    return NEXT_PREFIX + feature_name


def add_predicted(feature_name: str):
    return PREDICTED_PREFIX + feature_name


def strip_predicted(feature_name: str):
    if feature_name.startswith(PREDICTED_PREFIX):
        return feature_name.lstrip(PREDICTED_PREFIX)
    return feature_name


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


def get_maybe_predicted(e: Dict, k: str):
    if k in e and add_predicted(k) in e:
        raise ValueError(f"ambiguous, dict has both {k} and {add_predicted(k)}")
    elif not (k in e or add_predicted(k) in e):
        raise ValueError(f"dict lacks both {k} and {add_predicted(k)}")
    elif k in e:
        return e[k]
    elif add_predicted(k) in e:
        return e[add_predicted(k)]
    else:
        raise RuntimeError()


def in_maybe_predicted(k: str, e: Dict):
    if k in e and add_predicted(k) in e:
        raise ValueError(f"ambiguous, dict has both {k} and {add_predicted(k)}")
    elif not (k in e or add_predicted(k) in e):
        return False
    return True


def use_gt_rope(example: Dict):
    example['rope'] = example['gt_rope']
    return example


def add_label(example: Dict, threshold: float):
    is_close = example['error'] < threshold
    example['is_close'] = tf.cast(is_close, dtype=tf.float32)


def tf_write_example(full_output_directory: pathlib.Path,
                     out_example: Dict,
                     example_idx: int):
    features = {k: float_tensor_to_bytes_feature(v) for k, v in out_example.items()}
    return tf_write_features(example_idx, features, full_output_directory)


def tf_write_features(example_idx, features, full_output_directory):
    record_filename = "example_{:09d}.tfrecords".format(example_idx)
    full_filename = full_output_directory / record_filename
    features['tfrecord_path'] = bytes_feature(full_filename.as_posix().encode("utf-8"))
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    example_str = example_proto.SerializeToString()
    record_options = tf.io.TFRecordOptions(compression_type='ZLIB')
    with tf.io.TFRecordWriter(str(full_filename), record_options) as writer:
        writer.write(example_str)
    return full_filename


def count_up_to_next_record_idx(full_output_directory):
    record_idx = 0
    while True:
        record_filename = "example_{:09d}.tfrecords".format(record_idx)
        full_filename = full_output_directory / record_filename
        if not full_filename.exists():
            break
        record_idx += 1
    return record_idx


