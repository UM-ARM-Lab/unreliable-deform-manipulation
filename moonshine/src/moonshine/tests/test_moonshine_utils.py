from unittest import TestCase

import numpy as np
import tensorflow as tf

from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_list_of_dicts, \
    dict_of_sequences_to_sequence_of_list_of_dicts_tf
from moonshine.tests import testing_utils


class Test(TestCase):
    def test_dict_of_sequences_to_sequence_of_list_of_dicts(self):
        d = {'a': [1, 2, 3]}
        expected = [{'a': 1}, {'a': 2}, {'a': 3}]
        out = dict_of_sequences_to_sequence_of_list_of_dicts(d, time_axis=0)
        testing_utils.assert_list_of_dicts_close_np(out, expected)

    def test_dict_of_sequences_to_sequence_of_list_of_dicts_time_dim(self):
        d = {'a': np.array([[1, 2, 3], [4, 5, 6]])}
        expected = [{'a': np.array([1, 4])}, {'a': np.array([2, 5])}, {'a': np.array([3, 6])}]
        out = dict_of_sequences_to_sequence_of_list_of_dicts_tf(d, time_axis=1)
        testing_utils.assert_list_of_dicts_close_np(out, expected)

    def test_dict_of_sequences_to_sequence_of_list_of_dicts_tf(self):
        d = {'a': tf.constant([1, 2, 3])}
        expected = [{'a': tf.constant([1])}, {'a': tf.constant([2])}, {'a': tf.constant([3])}]
        out = dict_of_sequences_to_sequence_of_list_of_dicts_tf(d, time_axis=0)
        testing_utils.assert_list_of_dicts_close_tf(out, expected)

    def test_dict_of_sequences_to_sequence_of_list_of_dicts_tf_time_dim(self):
        d = {'a': tf.constant([[1, 2, 3], [4, 5, 6]])}
        expected = [{'a': tf.constant([1, 4])}, {'a': tf.constant([2, 5])}, {'a': tf.constant([3, 6])}]
        out = dict_of_sequences_to_sequence_of_list_of_dicts_tf(d, time_axis=1)
        testing_utils.assert_list_of_dicts_close_tf(out, expected)
