from unittest import TestCase

import tensorflow as tf

from link_bot_data.recovery_actions_utils import is_recovering, recovering_mask, starts_recovering
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


class Test(TestCase):
    def test_is_recovering(self):
        is_close = tf.constant([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 1],
        ], dtype=tf.float32)
        out = is_recovering(is_close)
        expected = tf.constant([
            False,
            True,
            True,
            True,
            False,
            False,
            False,
        ], dtype=tf.bool)
        self.assertTrue(tf.reduce_all(tf.equal(out, expected)))

    def test_starts_recovering(self):
        is_close = tf.constant([1, 0, 1, 1, 1], dtype=tf.float32)
        out = starts_recovering(is_close)
        self.assertTrue(out)

        is_close = tf.constant([1, 0, 1, 1, 0], dtype=tf.float32)
        out = starts_recovering(is_close)
        self.assertTrue(out)

        is_close = tf.constant([1, 0, 0, 0, 0], dtype=tf.float32)
        out = starts_recovering(is_close)
        self.assertFalse(out)

    def test_recovering_mask(self):
        is_close = tf.constant([[1, 0, 1, 1, 0, 0],
                                [1, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0],
                                [1, 0, 0, 0, 0, 0],
                                [1, 0, 1, 1, 1, 0],
                                [1, 0, 1, 0, 0, 1],
                                [1, 0, 0, 1, 0, 0],
                                [1, 1, 1, 1, 1, 1],
                                [1, 1, 0, 0, 0, 0],
                                [1, 0, 1, 0, 1, 0]],
                               dtype=tf.float32)
        out = recovering_mask(is_close)
        expected = tf.constant([[True, True, True, False, False],
                                [True, True, True, True, True],
                                [True, True, True, True, False],
                                [False, False, False, False, False],
                                [True, True, True, True, False],
                                [True, True, False, False, False],
                                [True, True, True, False, False],
                                [False, False, False, False, False],
                                [False, False, False, False, False],
                                [True, True, False, False, False]],
                               dtype=tf.bool)
        self.assertTrue(tf.reduce_all(tf.equal(out, expected)))
