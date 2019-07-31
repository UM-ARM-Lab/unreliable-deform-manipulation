#!/usr/bin/env python
from __future__ import print_function, division

import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_vector_feature(values):
    """Returns a float_list from 1-dimensional numpy array"""
    assert values.ndim == 1
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def serialize_example_pyfunction(images_traj, states_traj, actions_traj):
    """ Creates a tf.Example message ready to be written to a file. """

    feature = {}
    time_steps = images_traj.shape[0]
    for t in range(time_steps):
        image_t_key = '{}/image_aux1/encoded'.format(t)
        state_t_key = '{}/endeffector_pos'.format(t)
        action_t_key = '{}/action'.format(t)

        # image = tf.io.serialize_tensor(images_traj[t]).numpy()
        image = images_traj[t]
        state = states_traj[t]
        action = actions_traj[t]

        feature[image_t_key] = _bytes_feature(image.numpy())
        feature[state_t_key] = _float_vector_feature(state.numpy())
        feature[action_t_key] = _float_vector_feature(action.numpy())

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(f0, f1, f2):
    tf_string = tf.py_function(serialize_example_pyfunction, (f0, f1, f2), tf.string)
    return tf.reshape(tf_string, ())


def serialize_example_pyfunction_4(images_traj, states_traj, actions_traj, constraints_traj):
    """ Creates a tf.Example message ready to be written to a file. """

    feature = {}
    time_steps = images_traj.shape[0]
    for t in range(time_steps):
        image_t_key = '{}/image_aux1/encoded'.format(t)
        state_t_key = '{}/endeffector_pos'.format(t)
        action_t_key = '{}/action'.format(t)
        constraint_t_key = '{}/constraint'.format(t)

        # image = tf.io.serialize_tensor(images_traj[t]).numpy()
        image = images_traj[t]
        state = states_traj[t]
        action = actions_traj[t]
        constraint = constraints_traj[t]

        feature[image_t_key] = _bytes_feature(image.numpy())
        feature[state_t_key] = _float_vector_feature(state.numpy())
        feature[action_t_key] = _float_vector_feature(action.numpy())
        feature[constraint_t_key] = _float_vector_feature(constraint.numpy())

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example_4(f0, f1, f2, f3):
    tf_string = tf.py_function(serialize_example_pyfunction_4, (f0, f1, f2, f3), tf.string)
    return tf.reshape(tf_string, ())
