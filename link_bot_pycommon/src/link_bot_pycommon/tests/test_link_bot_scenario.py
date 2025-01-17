from unittest import TestCase

import tensorflow as tf

from link_bot_pycommon.link_bot_scenario import LinkBotScenario
from moonshine.tests.testing_utils import assert_close_tf


class TestLinkBotScenario(TestCase):
    def test_to_rope_local_frame(self):
        rope_states = tf.constant([[0, 0, 1, 0, 1, 1],
                                   [2, 2, 2, 1, 3, 1],
                                   [-2, -2, -2, -1, -3, -1],
                                   [1, 0, 1, 1, 2, 1]],
                                  dtype=tf.float32)
        expected_states = tf.constant([[-1, 1, -1, 0, 0, 0],
                                       [-1, 1, -1, 0, 0, 0],
                                       [-1, 1, -1, 0, 0, 0],
                                       [-1, -1, -1, 0, 0, 0]],
                                      dtype=tf.float32)
        out_states = LinkBotScenario.to_rope_local_frame_tf(rope_states)
        assert_close_tf(out_states, expected_states, atol=1e-5)

    def test_to_rope_local_frame_with_reference(self):
        rope_states = tf.constant([[0, 0, 1, 0, 1, 1]], dtype=tf.float32)
        reference_rope_states = tf.constant([[0, 0, 1, 0, 2, 0]], dtype=tf.float32)
        expected_states = tf.constant([[-2, 0, -1, 0, -1, 1]], dtype=tf.float32)
        out_states = LinkBotScenario.to_rope_local_frame_tf(rope_states, reference_rope_states)
        assert_close_tf(out_states, expected_states, atol=1e-5)
