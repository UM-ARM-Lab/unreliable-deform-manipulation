from unittest import TestCase

import tensorflow as tf

from moonshine.matrix_operations import batch_outer_product
from moonshine.tests.testing_utils import assert_close_tf


class Test(TestCase):
    def test_batch_outer_product(self):
        a = tf.constant([[1, 2], [0, 2]], dtype=tf.float32)
        b = tf.constant([[3, 4, 5], [1, 4, 2]], dtype=tf.float32)
        out = batch_outer_product(a, b)
        expected = tf.constant([[[3, 4, 5], [6, 8, 10]], [[0, 0, 0], [2, 8, 4]]], dtype=tf.float32)
        assert_close_tf(out, expected)
