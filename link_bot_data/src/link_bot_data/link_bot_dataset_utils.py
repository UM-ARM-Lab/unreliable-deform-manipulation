#!/usr/bin/env python
from __future__ import print_function, division

import os
import pathlib
from typing import Optional, Dict, List

import git
import tensorflow as tf
from colorama import Fore

from link_bot_pycommon import link_bot_pycommon


def parse_and_deserialize(dataset, feature_description, n_parallel_calls=None):
    parsed_dataset = parse_dataset(dataset, feature_description, n_parallel_calls=n_parallel_calls)
    deserialized_dataset = deserialize(parsed_dataset, n_parallel_calls=n_parallel_calls)
    return deserialized_dataset


def parse_dataset(dataset, feature_description, n_parallel_calls=None):
    # @tf.function
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

    # @tf.function
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
    # @tf.function
    def _label_is(label_is):
        # @tf.function
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


# @tf.function
def convert_sequences_to_transitions(constant_data: Dict,
                                     state_like_sequences: Dict,
                                     action_like_sequences: Dict,
                                     planned_state_keys: List[str]):
    # Create a dict of lists, where keys are the features we want in each transition, and values are the data.
    # The first dimension of these values is what will be split up into different examples
    transitions = {
    }

    for feature_name in state_like_sequences.keys():
        next_feature_name = add_next(feature_name)
        transitions[feature_name] = []
        transitions[next_feature_name] = []

    for feature_name in planned_state_keys:
        transitions[add_all(add_planned(feature_name))] = []

    for feature_name in action_like_sequences.keys():
        transitions[feature_name] = []
        transitions[add_all(feature_name)] = []

    for feature_name in constant_data.keys():
        transitions[feature_name] = []

    # @tf.function
    def _null_pad_sequence(sequence, idx):
        # this should be some number that will be very far from being inside any actual
        # because we're gonna try to draw it anyways
        null = -10000
        if idx + 2 < sequence.shape[0]:
            sequence[idx + 2:] = null
        return sequence

    # Fill the transitions dictionary with the data from the sequences
    sequence_length = action_like_sequences['action'].shape[0]
    for transition_idx in range(sequence_length):
        for feature_name in state_like_sequences.keys():
            transitions[feature_name].append(state_like_sequences[feature_name][transition_idx])

            next_feature_name = add_next(feature_name)
            transitions[next_feature_name].append(state_like_sequences[feature_name][transition_idx + 1])

        for feature_name in planned_state_keys:
            planned_feature_name = add_planned(feature_name)
            planned_state_sequence = state_like_sequences[planned_feature_name]
            null_pad_args = [planned_state_sequence, transition_idx]
            null_padded_planned_state_sequence = tf.numpy_function(_null_pad_sequence, null_pad_args, tf.float32)
            null_padded_planned_state_sequence.set_shape(state_like_sequences[planned_feature_name].shape)
            transitions[add_all(planned_feature_name)].append(null_padded_planned_state_sequence)

        for feature_name in action_like_sequences.keys():
            transitions[feature_name].append(action_like_sequences[feature_name][transition_idx])

            action_sequence = action_like_sequences[feature_name]
            null_pad_args = [action_sequence, transition_idx]
            null_padded_planned_state_sequence = tf.numpy_function(_null_pad_sequence, null_pad_args, tf.float32)
            null_padded_planned_state_sequence.set_shape(action_like_sequences[feature_name].shape)
            transitions[add_all(feature_name)].append(null_padded_planned_state_sequence)

        for feature_name in constant_data.keys():
            transitions[feature_name].append(constant_data[feature_name])

    transition_dataset = tf.data.Dataset.from_tensor_slices(transitions)
    return transition_dataset
