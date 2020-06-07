from unittest import TestCase

import tensorflow as tf

from link_bot_data.recovery_actions_utils import is_recovering


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
