#!/usr/bin/env python
from __future__ import print_function, division

import os
import pathlib

from typing import Optional

import git
import tensorflow as tf
from colorama import Fore

from link_bot_pycommon import link_bot_pycommon
from moonshine.numpy_utils import add_batch
from moonshine.raster_points_layer import make_transition_image, make_traj_image


def parse_and_deserialize(dataset, feature_description, n_parallel_calls=None):
    # NOTE: assumes all features are floats
    def _parse(example_proto):
        deserialized_dict = tf.io.parse_single_example(example_proto, feature_description)
        return deserialized_dict

    # the elements of parsed dataset are dictionaries with the serialized tensors as strings
    parsed_dataset = dataset.map(_parse, num_parallel_calls=n_parallel_calls)

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


def balance(dataset, label_key='label'):
    def _label_is(label_is):
        def __filter(transition):
            result = tf.squeeze(tf.equal(transition[label_key], label_is))
            return result

        return __filter

    # In order to figure out whether the are fewer negative or positive examples,
    # we iterate over the first `min_test_examples` elements. If after this many
    # we see a clear imbalance in one direction, make the decision then.
    # Otherwise, we keep iterating up until `max_test_examples`, checking until we 
    # see a clear imabalance. This is nessecary because iterating over the whole thing (even once)
    # is very slow (can take minutes), and most imbalanced datasets are obviously imbalanced
    # so we need not check every example
    positive_examples = 0
    negative_examples = 0
    examples_considered = 0
    min_test_examples = 10
    max_test_examples = 100
    margin = 2
    for examples_considered, example in enumerate(dataset):
        if examples_considered > max_test_examples:
            break
        if examples_considered > min_test_examples:
            if positive_examples > negative_examples + margin:
                break
            elif negative_examples > positive_examples + margin:
                break
        if tf.equal(tf.squeeze(example[label_key]), 1):
            positive_examples += 1
        else:
            negative_examples += 1
    fewer_negative = negative_examples < positive_examples
    print("considered {} elements. found {} positive, {} negative".format(examples_considered, positive_examples,
                                                                          negative_examples))

    if fewer_negative:
        positive_examples = dataset.filter(_label_is(1))

        negative_examples = dataset.filter(_label_is(0))
        negative_examples = negative_examples.cache(cachename())
        negative_examples = negative_examples.repeat()

        balanced_dataset = tf.data.Dataset.zip((positive_examples, negative_examples))
    else:
        negative_examples = dataset.filter(_label_is(0))

        positive_examples = dataset.filter(_label_is(1))
        positive_examples = positive_examples.cache(cachename())
        positive_examples = positive_examples.repeat()

        balanced_dataset = tf.data.Dataset.zip((negative_examples, positive_examples))

    balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)

    return balanced_dataset


def add_traj_image(dataset):
    def _add_traj_image(input_dict):
        full_env = input_dict['full_env/env']
        full_env_origin = input_dict['full_env/origin']
        res = input_dict['res']
        stop_index = input_dict['planned_state/link_bot_all_stop']
        planned_states = input_dict['planned_state/link_bot_all'][:stop_index]

        h, w = full_env.shape

        image = tf.numpy_function(make_traj_image, add_batch(full_env, full_env_origin, res, planned_states), tf.float32)[0]
        image.set_shape([h, w, 3])

        input_dict['trajectory_image'] = image
        return input_dict

    return dataset.map(_add_traj_image)


def add_transition_image(dataset, action_in_image: Optional[bool] = False):
    def _add_transition_image(input_dict):
        """
        Expected sizes:
            'action': n_action
            'planned_state': n_state
            'planned_state_next': n_state
            'planned_local_env/env': h, w
            'planned_local_env/origin': 2
            'planned_local_env/extent': 4
            'resolution': 1
        """
        action = input_dict['action']
        planned_local_env = input_dict['planned_state/local_env']
        res = input_dict['res']
        origin = input_dict['planned_state/local_env_origin']
        planned_state = input_dict['planned_state/link_bot']
        planned_next_state = input_dict['planned_state_next/link_bot']
        n_action = action.shape[0]
        h, w = planned_local_env.shape
        n_points = link_bot_pycommon.n_state_to_n_points(planned_state.shape[0])

        image = tf.numpy_function(make_transition_image,
                                  [planned_local_env, planned_state, action, planned_next_state, res, origin, action_in_image],
                                  tf.float32)
        image.set_shape([h, w, 1 + n_points + n_points + n_action])

        input_dict['transition_image'] = image
        return input_dict

    return dataset.map(_add_transition_image)


def cachename(mode: Optional[str] = None):
    if mode is not None:
        tmpname = "/tmp/tf_{}_{}".format(mode, link_bot_pycommon.rand_str())
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
