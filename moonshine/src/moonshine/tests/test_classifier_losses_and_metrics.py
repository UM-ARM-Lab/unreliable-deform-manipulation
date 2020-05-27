from unittest import TestCase

import tensorflow as tf

from moonshine.classifier_losses_and_metrics import reconverging_weighted_binary_classification_sequence_loss_function, \
    negative_weighted_binary_classification_sequence_loss_function
from moonshine.gpu_config import limit_gpu_mem
from moonshine.tests.testing_utils import assert_close_tf

limit_gpu_mem(0.1)


class Test(TestCase):
    def test_reconverging_weighted_binary_classification_sequence_loss_function_correct(self):
        data = {
            'is_close': tf.constant([[1, 1, 1], [1, 1, 0], [1, 0, 1]], tf.float32)
        }
        pred = {
            'logits': tf.constant([[[100], [100]], [[100], [-100]], [[-100], [100]]], tf.float32),
            'mask': tf.constant([[True, True, True], [True, True, True], [True, True, True]], tf.bool),
        }
        expected_loss = tf.constant([0.0])
        out_loss = reconverging_weighted_binary_classification_sequence_loss_function(data, pred)
        assert_close_tf(expected_loss, out_loss)

    def test_reconverging_weighted_binary_classification_sequence_loss_function_incorrect(self):
        data = {
            'is_close': tf.constant([[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1]], tf.float32)
        }
        pred = {
            'logits': tf.constant([[[100], [-100]], [[100], [-100]], [[100], [-100]], [[-100], [-100]]], tf.float32),
            'mask': tf.constant([[True, True, False], [True, True, True], [True, True, True], [True, True, True]], tf.bool),
        }
        expected_loss = tf.constant([28.571428])
        out_loss = reconverging_weighted_binary_classification_sequence_loss_function(data, pred)
        assert_close_tf(expected_loss, out_loss)

    def test_negative_weighted_binary_classification_sequence_loss_function_correct(self):
        data = {
            'is_close': tf.constant([[1, 1, 1], [1, 1, 0], [1, 0, 1]], tf.float32)
        }
        pred = {
            'logits': tf.constant([[[100], [100]], [[100], [-100]], [[-100], [100]]], tf.float32),
            'mask': tf.constant([[True, True, True], [True, True, True], [True, True, True]], tf.bool),
        }
        expected_loss = tf.constant([0.0])
        out_loss = negative_weighted_binary_classification_sequence_loss_function(data, pred)
        assert_close_tf(expected_loss, out_loss)

    def test_negative_weighted_binary_classification_sequence_loss_function_incorrect(self):
        data = {
            'is_close': tf.constant([[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1]], tf.float32)
        }
        pred = {
            'logits': tf.constant([[[100], [-100]], [[100], [-100]], [[100], [-100]], [[-100], [-100]]], tf.float32),
            'mask': tf.constant([[True, True, False], [True, True, True], [True, True, True], [True, True, True]], tf.bool),
        }
        expected_loss = tf.constant([9.52381])
        out_loss = negative_weighted_binary_classification_sequence_loss_function(data, pred)
        assert_close_tf(expected_loss, out_loss)
