#!/usr/bin/env python
from __future__ import print_function, division

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


def tf_serialize_example(images, states, actions):
    tf_string = tf.py_function(serialize_example_pyfunction, (images, states, actions), tf.string)
    return tf.reshape(tf_string, ())
