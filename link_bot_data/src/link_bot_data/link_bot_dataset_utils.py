#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
import random
from typing import Optional
import tensorflow as tf
from link_bot_pycommon import link_bot_pycommon
from moonshine.raster_points_layer import RasterPoints, raster


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


def balance_by_augmentation(dataset, image_key, label_key):
    """
    generate more examples by random 90 rotations or horizontal/vertical flipping
    """
    import time
    def _label_is(label_is):
        def __filter(transition):
            result = tf.squeeze(tf.equal(transition[label_key], label_is))
            return result

        return __filter

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
        image = input_dict[image_key]
        r = tf.random.uniform([], 0, 6, dtype=tf.int64)
        augmented_image = tf.numpy_function(_augment, inp=[r, image], Tout=tf.float32)

        input_dict[image_key] = augmented_image
        return  input_dict

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
    min_test_examples = 100
    max_test_examples = 1000
    margin = 10
    for examples_considered, example in enumerate(dataset):
        if examples_considered > max_test_examples:
            break
        if examples_considered > min_test_examples:
            if positive_examples > negative_examples + margin:
                fewer_negative = True
                break
            elif negative_examples > positive_examples + margin:
                fewer_negative = False
                break
        if tf.equal(tf.squeeze(example[label_key]), 1):
            positive_examples += 1
        else:
            negative_examples += 1
    fewer_negative = negative_examples < positive_examples
    print("considered {} elements. found {} positive, {} negative".format(examples_considered, positive_examples, negative_examples))

    if fewer_negative:
        positive_examples = dataset.filter(_label_is(1))

        negative_examples = dataset.filter(_label_is(0))
        negative_examples = negative_examples.cache(cachename())
        negative_examples = negative_examples.repeat()

        augmented_negative_examples = negative_examples.map(augment)

        balanced_dataset = tf.data.Dataset.zip((positive_examples, augmented_negative_examples))
    else:
        negative_examples = dataset.filter(_label_is(0))

        positive_examples = dataset.filter(_label_is(1))
        positive_examples = positive_examples.cache(cachename())
        positive_examples = positive_examples.repeat()

        augmented_positive_examples = positive_examples.map(augment)

        balanced_dataset = tf.data.Dataset.zip((negative_examples, augmented_positive_examples))

    balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)

    return balanced_dataset

# raster each state into an image
def raster_rope_images(planned_states, res, h, w, n_points):
    # NOTE: assumes full env is centered at origin
    origin = [int(w/2), int(h/2)]
    rope_images = np.zeros([h, w, n_points], dtype=np.float32)
    for planned_state in planned_states:
        rope_img_t = raster(planned_state, res, origin, h, w)
        # NOTE: consider appending channel-wise instead?
        rope_images += rope_img_t
    return rope_images

def add_traj_image(input_dict):
    # TODO: incorporate action somehow? maybe one channel per action per time step
    full_env = input_dict['full_env/env']
    res = input_dict['res']
    stop_index = input_dict['planned_state/link_bot_all_stop']
    planned_states = input_dict['planned_state/link_bot_all'][:stop_index]

    h, w = full_env.shape
    t = planned_states.shape[0]
    n_points = link_bot_pycommon.n_state_to_n_points(planned_states.shape[1])

    # add channel index
    full_env = tf.expand_dims(full_env, axis=2)


    rope_imgs = tf.numpy_function(raster_rope_images, [planned_states, res, h, w, n_points], tf.float32)
    rope_imgs.set_shape([h, w, n_points])

    # h, w, channel
    image = tf.concat((full_env, rope_imgs), axis=2)
    input_dict['trajectory_image'] = image
    return input_dict

def add_transition_image(input_dict):
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
    local_env = input_dict['planned_state/local_env']
    res = input_dict['res']
    origin = input_dict['planned_state/local_env_origin']
    planned_state = input_dict['planned_state/link_bot']
    planned_next_state = input_dict['planned_state_next/link_bot']
    h, w = local_env.shape
    n_points = link_bot_pycommon.n_state_to_n_points(planned_state.shape[0])

    # add channel index
    planned_local_env = tf.expand_dims(local_env, axis=2)

    # raster each state into an image
    planned_rope_image = tf.numpy_function(raster, [planned_state, res, origin, h, w], tf.float32)
    planned_rope_image.set_shape([h, w, n_points])
    planned_next_rope_image = tf.numpy_function(raster, [planned_next_state, res, origin, h, w], tf.float32)
    planned_next_rope_image.set_shape([h, w, n_points])

    # action
    # add spatial dimensions and tile
    action_reshaped = tf.expand_dims(tf.expand_dims(action, axis=0), axis=0)
    action_image = tf.tile(action_reshaped, [h, w, 1], name='action_spatial_tile')

    # h, w, channel
    image = tf.concat((planned_rope_image, planned_next_rope_image, planned_local_env, action_image), axis=2)
    input_dict['transition_image'] = image
    return input_dict

def cachename(mode : Optional[str] = None):
    if mode is not None:
        tmpname = "/tmp/tf_{}_{}".format(mode, link_bot_pycommon.rand_str())
    else:
        tmpname = "/tmp/tf_{}".format(link_bot_pycommon.rand_str())
    return tmpname
