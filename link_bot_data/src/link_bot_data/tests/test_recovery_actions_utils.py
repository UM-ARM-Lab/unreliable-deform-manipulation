from unittest import TestCase

import tensorflow as tf

from link_bot_data.recovery_actions_utils import recovering_mask
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(0.1)


class Test(TestCase):
    def test_recovering_mask(self):
        is_close = tf.constant([[1, 0, 0, 1, 1],
                                [1, 0, 0, 0, 0],
                                [1, 1, 1, 1, 0],
                                [1, 1, 1, 0, 1],
                                [1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 1],
                                [1, 0, 1, 1, 0],
                                [1, 1, 0, 1, 1],
                                [0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 1],
                                [1, 0, 1, 0, 1]],
                               dtype=tf.int32)
        out = recovering_mask(is_close)
        expected = tf.constant([[True, True, False, False, False],
                                [True, True, False, False, False],
                                [True, True, True, True, True],
                                [True, True, True, True, False],
                                [False, False, False, False, False],
                                [True, True, False, False, False],
                                [True, True, False, False, False],
                                [True, True, True, False, False],
                                [False, False, False, False, False],
                                [False, False, False, False, False],
                                [True, True, False, False, False]],
                               dtype=tf.bool)
        self.assertTrue(tf.reduce_all(tf.equal(out, expected)))
