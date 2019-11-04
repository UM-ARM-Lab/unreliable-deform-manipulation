#!/usr/bin/env python
from __future__ import print_function, division

import json
from typing import Union, Optional

import tensorflow as tf

from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset
from video_prediction.datasets import get_dataset_class


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


def serialize_example_pyfunction(images_traj, states_traj, actions_traj):
    # TODO: there's duplicate information here that could be remove. This could take in a dictionary
    # and the type/name/shape information could be set by a feature description dictionary or something in
    # the file that's calling this function
    feature = {}
    time_steps = images_traj.shape[0]
    for t in range(time_steps):
        image_t_key = '{}/image_aux1/encoded'.format(t)
        state_t_key = '{}/endeffector_pos'.format(t)
        action_t_key = '{}/action'.format(t)

        image = images_traj[t]
        state = states_traj[t]
        action = actions_traj[t]

        feature[image_t_key] = bytes_feature(image.numpy())
        feature[state_t_key] = float_feature(state.numpy())
        feature[action_t_key] = float_feature(action.numpy())

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(images, states, actions):
    tf_string = tf.py_function(serialize_example_pyfunction, (images, states, actions), tf.string)
    return tf.reshape(tf_string, ())


def flatten_concat_pairs(ex_pos, ex_neg):
    flat_pair = tf.data.Dataset.from_tensors(ex_pos).concatenate(tf.data.Dataset.from_tensors(ex_neg))
    return flat_pair


def balance_x_dataset(dataset, key, fewer_negative=True):
    """
    :param dataset: assumes each element is of the structure (inputs_dict_of_tensors, outputs_dict_of_tensors)
    :param key: string to index into y component of dataset, used to determine labels
    :param fewer_negative: fewer negative examples than positive
    """
    if fewer_negative:
        negative_examples = dataset.filter(lambda x: tf.squeeze(tf.equal(x[key], 0))).repeat()
        positive_examples = dataset.filter(lambda x: tf.squeeze(tf.equal(x[key], 1)))
    else:
        positive_examples = dataset.filter(lambda x: tf.squeeze(tf.equal(x[key], 1))).repeat()
        negative_examples = dataset.filter(lambda x: tf.squeeze(tf.equal(x[key], 0)))

    # zipping takes the shorter of the two, hence why this makes it balanced
    balanced_dataset = tf.data.Dataset.zip((positive_examples, negative_examples))
    balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)
    return balanced_dataset


def balance_xy_dataset(dataset, key):
    """
    :param dataset: assumes each element is of the structure (inputs_dict_of_tensors, outputs_dict_of_tensors)
    :param key: string to index into y component of dataset, used to determine labels
    :param fewer_negative: fewer negative examples than positive
    """
    negative_examples = dataset.filter(lambda x, y: tf.squeeze(tf.equal(y[key], 0)))
    positive_examples = dataset.filter(lambda x, y: tf.squeeze(tf.equal(y[key], 1)))

    print(len(positive_examples), len(negative_examples))

    # zipping takes the shorter of the two, hence why this makes it balanced
    balanced_dataset = tf.data.Dataset.zip((positive_examples, negative_examples))
    balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)
    return balanced_dataset


# TODO: deduplicate with video_prediction
def get_dataset(dataset_directory: str,
                dataset_hparams_dict: Union[str, dict],
                dataset_hparams: str,
                mode: str,
                epochs: Optional[int],
                batch_size: int,
                seed: int,
                balance_key: Optional[str] = None,
                shuffle: bool = True):
    if isinstance(dataset_hparams_dict, str):
        dataset_hparams_dict = json.load(open(dataset_hparams_dict, 'r'))

    my_dataset = LinkBotStateSpaceDataset(dataset_directory,
                                          mode=mode,
                                          num_epochs=epochs,
                                          seed=seed,
                                          hparams_dict=dataset_hparams_dict,
                                          hparams=dataset_hparams)

    if balance_key is not None:
        tf_dataset = my_dataset.make_dataset(batch_size=batch_size)
        tf_dataset = balance_xy_dataset(tf_dataset, balance_key)
        tf_dataset = tf_dataset.batch(batch_size)
    else:
        tf_dataset = my_dataset.make_dataset(batch_size, shuffle=shuffle)

    return my_dataset, tf_dataset
