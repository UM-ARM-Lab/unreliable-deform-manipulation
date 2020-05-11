from unittest import TestCase
import numpy as np
import tensorflow as tf

from moonshine.tests import testing_utils


class Test(TestCase):
    def test_assert_dicts_close_np(self):
        a = {
            'x': np.random.randn(3),
            'y': np.random.randn(3)
        }
        b = {
            'x': np.random.randn(3),
            'y': np.random.randn(3)
        }
        testing_utils.assert_dicts_close_np(a, a)
        with self.assertRaises(AssertionError):
            testing_utils.assert_dicts_close_np(a, b)

    def test_assert_dicts_close_tf(self):
        a = {
            'x': tf.random.normal([3]),
            'y': tf.random.normal([3])
        }
        b = {
            'x': tf.random.normal([3]),
            'y': tf.random.normal([3])
        }
        testing_utils.assert_dicts_close_tf(a, a)
        with self.assertRaises(AssertionError):
            testing_utils.assert_dicts_close_tf(a, b)

    def test_assert_list_of_dicts_close_np(self):
        a = [
            {
                'x': np.random.randn(3),
                'y': np.random.randn(3)
            },
            {
                'x': np.random.randn(3),
                'y': np.random.randn(3)
            }
        ]
        b = [
            {
                'x': np.random.randn(3),
                'y': np.random.randn(3)
            },
            {
                'x': np.random.randn(3),
                'y': np.random.randn(3)
            }
        ]
        # should NOT raise
        testing_utils.assert_list_of_dicts_close_np(a, a)
        # SHOULD raise
        with self.assertRaises(AssertionError):
            testing_utils.assert_list_of_dicts_close_np(a, b)

    def test_assert_list_of_dicts_close_tf(self):
        a = [
            {
                'x': tf.random.normal([3]),
                'y': tf.random.normal([3])
            },
            {
                'x': tf.random.normal([3]),
                'y': tf.random.normal([3])
            }
        ]
        b = [
            {
                'x': tf.random.normal([3]),
                'y': tf.random.normal([3])
            },
            {
                'x': tf.random.normal([3]),
                'y': tf.random.normal([3])
            }
        ]
        # should NOT raise
        testing_utils.assert_list_of_dicts_close_tf(a, a)
        # SHOULD raise
        with self.assertRaises(AssertionError):
            testing_utils.assert_list_of_dicts_close_tf(a, b)
