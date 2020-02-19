#!/usr/bin/env python
from __future__ import print_function, division

import tensorflow as tf


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


def balance_by_augmentation(dataset, key):
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

        input_dict['image'] = augmented_image
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
        if tf.equal(tf.squeeze(example[key]), 1):
            positive_examples += 1
        else:
            negative_examples += 1
    fewer_negative = negative_examples < positive_examples
    print("consisdered {} elements. found {} positive, {} negative".format(examples_considered, positive_examples, negative_examples))

    if fewer_negative:
        positive_examples = dataset.filter(_label_is(1))

        n_positive_examples = 0
        for _ in positive_examples:
            n_positive_examples += 1

        negative_examples = dataset.filter(_label_is(0)).repeat().take(n_positive_examples)

        augmented_negative_examples = negative_examples.map(augment)

        balanced_dataset = tf.data.Dataset.zip((positive_examples, augmented_negative_examples))
    else:
        negative_examples = dataset.filter(_label_is(0))

        n_negative_examples = 0
        for _ in negative_examples:
            n_negative_examples += 1

        positive_examples = dataset.filter(_label_is(1)).repeat().take(n_negative_examples)

        augmented_positive_examples = positive_examples.map(augment)

        balanced_dataset = tf.data.Dataset.zip((negative_examples, augmented_positive_examples))

    balanced_dataset = balanced_dataset.flat_map(flatten_concat_pairs)

    return balanced_dataset
