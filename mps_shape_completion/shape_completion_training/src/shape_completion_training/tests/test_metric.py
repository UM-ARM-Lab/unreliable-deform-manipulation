#! /usr/bin/env python

import unittest
import tensorflow as tf
import numpy as np
from shape_completion_training import metric


class TestMetrics(unittest.TestCase):
    def test_metrics(self):
        y_true = tf.constant([0, 1, 0, 0, 1, 1], tf.float32)
        y_pred = tf.constant([1, 0, 1, 0, 1, 1], tf.float32)

        self.assertEqual(metric.fp(y_true=y_true, y_pred=y_pred), 2)
        self.assertEqual(metric.fn(y_true=y_true, y_pred=y_pred), 1)
        self.assertEqual(metric.tp(y_true=y_true, y_pred=y_pred), 2)
        self.assertEqual(metric.tn(y_true=y_true, y_pred=y_pred), 1)


if __name__ == '__main__':
    unittest.main()
