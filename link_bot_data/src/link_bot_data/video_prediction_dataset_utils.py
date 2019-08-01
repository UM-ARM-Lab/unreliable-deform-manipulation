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


def serialize_example_pyfunction(images_traj, states_traj, actions_traj, constraints_traj=None, sdf_traj=None):
    # TODO: there's duplicate information here that could be remove. This could take in a dictionary
    # and the type/name/shape information could be set by a feature description dictionary or something in
    # the file that's calling this function
    feature = {}
    time_steps = images_traj.shape[0]
    for t in range(time_steps):
        image_t_key = '{}/image_aux1/encoded'.format(t)
        state_t_key = '{}/endeffector_pos'.format(t)
        action_t_key = '{}/action'.format(t)
        sdf_t_key = '{}/sdf'.format(t)
        constraint_t_key = '{}/constraint'.format(t)

        image = images_traj[t]
        state = states_traj[t]
        sdf = sdf_traj[t]
        action = actions_traj[t]
        if constraints_traj is not None:
            constraint = constraints_traj[t]

        feature[image_t_key] = _bytes_feature(image.numpy())
        feature[state_t_key] = _float_vector_feature(state.numpy())
        feature[action_t_key] = _float_vector_feature(action.numpy())
        if constraints_traj is not None:
            feature[constraint_t_key] = _float_vector_feature(constraint.numpy())
        if sdf_traj is not None:
            feature[sdf_t_key] = _bytes_feature(sdf.numpy())

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(images, states, actions, constraints=None):
    tf_string = tf.py_function(serialize_example_pyfunction, (images, states, actions, constraints), tf.string)
    return tf.reshape(tf_string, ())
