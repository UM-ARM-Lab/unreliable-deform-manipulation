#!/usr/bin/env python
from __future__ import print_function, division

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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


@tf.function
def tf_serialize_example(images, states, actions):
    tf_string = tf.py_function(serialize_example_pyfunction, (images, states, actions), tf.string)
    return tf.reshape(tf_string, ())


def flatten_concat_pairs(ex_pos, ex_neg):
    flat_pair = tf.data.Dataset.from_tensors(ex_pos).concatenate(tf.data.Dataset.from_tensors(ex_neg))
    return flat_pair


def balance_dataset(dataset, key, fewer_negative=True):
    """
    Throw out examples of whichever dataset has more.
    :param dataset: assumes each element is of the structure (inputs_dict_of_tensors, outputs_dict_of_tensors)
    :param key: string to index into y component of dataset, used to determine labels
    :param fewer_negative: fewer negative examples than positive
    """

    def _label_is(label_is):
        def __filter(transition):
            return tf.squeeze(tf.equal(transition[key], label_is))

        return __filter

    if fewer_negative:
        negative_examples = dataset.filter(_label_is(0)).repeat()
        positive_examples = dataset.filter(_label_is(1))
    else:
        positive_examples = dataset.filter(_label_is(1)).repeat()
        negative_examples = dataset.filter(_label_is(0))

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

    # zipping takes the shorter of the two, hence why this makes it balanced
    balanced_dataset = tf.data.Dataset.zip((positive_examples, negative_examples))
    balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)
    return balanced_dataset


def balance_by_augmentation(dataset, key, fewer_negative=True):
    """
    generate more examples by random 90 rotations or horizontal/vertical flipping
    :param dataset:
    :param key:
    :return:
    """

    def _label_is(label_is):
        def __filter(transition):
            return tf.squeeze(tf.equal(transition[key], label_is))

        return __filter

    def _debug(image, augmented_image):
        fig, axes = plt.subplots(2)
        axes[0].imshow(np.sum(image, axis=3).squeeze())
        axes[1].imshow(np.sum(augmented_image, axis=3).squeeze())
        i = np.random.randint(0, 1000)
        plt.savefig('/tmp/{}.png'.format(i))

    def _augment(r, image):
        if r == 1:
            augmented_image = tf.image.rot90(tf.image.rot90(tf.image.rot90(image)))
        elif r == 2:
            augmented_image = tf.image.rot90(tf.image.rot90(image))
        elif r == 3:
            augmented_image = tf.image.rot90(image)
        elif r == 4:
            augmented_image = tf.image.flip_up_down(image)
        else:
            augmented_image = tf.image.flip_left_right(image)
        return augmented_image

    def augment(input_dict):
        image = input_dict['image']
        r = tf.random.uniform([], 0, 6, dtype=tf.int64)
        augmented_image = tf.numpy_function(_augment, inp=[r, image], Tout=tf.float32)

        return {
            'image': augmented_image,
            'label': input_dict['label']
        }

    if fewer_negative:
        positive_examples = dataset.filter(_label_is(1))

        n_positive_examples = 0
        for _ in positive_examples:
            n_positive_examples += 1

        negative_examples = dataset.filter(_label_is(0)).repeat().take(n_positive_examples)

        augmented_negative_examples = negative_examples.map(augment)

        # zipping takes the shorter of the two, hence why this makes it balanced
        balanced_dataset = tf.data.Dataset.zip((positive_examples, augmented_negative_examples))
    else:
        raise NotImplementedError()  # TODO: copy the above branch but for positive examples

    balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)

    return balanced_dataset
