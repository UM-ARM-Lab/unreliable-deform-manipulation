import unittest

import numpy as np
import tensorflow as tf

from link_bot_data.dataset_utils import is_reconverging, null_pad, NULL_PAD_VALUE, num_reconverging, \
    num_reconverging_subsequences, add_predicted, remove_predicted, remove_predicted_from_dict
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import remove_batch

limit_gpu_mem(0.1)


class MyTestCase(unittest.TestCase):
    def test_is_reconverging(self):
        batch_is_reconverging_output = is_reconverging(tf.constant([[1, 0, 0, 1], [1, 1, 1, 0], [1, 0, 0, 0]], tf.int64)).numpy()
        self.assertTrue(batch_is_reconverging_output[0])
        self.assertFalse(batch_is_reconverging_output[1])
        self.assertFalse(batch_is_reconverging_output[2])
        self.assertTrue(remove_batch(is_reconverging(tf.constant([[1, 0, 0, 1]], tf.int64))).numpy())
        self.assertFalse(remove_batch(is_reconverging(tf.constant([[1, 0, 0, 0]], tf.int64))).numpy())
        self.assertFalse(remove_batch(is_reconverging(tf.constant([[1, 0, 1, 0]], tf.int64))).numpy())

    def test_num_reconverging_subsequences(self):
        self.assertEqual(num_reconverging_subsequences(tf.constant([[1, 0, 0, 1], [1, 1, 1, 0], [1, 0, 1, 1]], tf.int64)).numpy(),
                         3)
        self.assertEqual(num_reconverging_subsequences(tf.constant([[1, 1, 0, 1, 1, 1]], tf.int64)).numpy(), 6)
        self.assertEqual(num_reconverging_subsequences(tf.constant([[1, 0, 0, 0]], tf.int64)).numpy(), 0)

    def test_num_reconverging(self):
        self.assertEqual(num_reconverging(tf.constant([[1, 0, 0, 1], [1, 1, 1, 0], [1, 0, 1, 1]], tf.int64)).numpy(), 2)
        self.assertEqual(num_reconverging(tf.constant([[1, 0, 0, 1]], tf.int64)).numpy(), 1)
        self.assertEqual(num_reconverging(tf.constant([[1, 0, 0, 0]], tf.int64)).numpy(), 0)

    def test_null_pad(self):
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=0, end=2),
                                   np.array([1, 0, 0, NULL_PAD_VALUE]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=0, end=3),
                                   np.array([1, 0, 0, 1]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=1, end=2),
                                   np.array([NULL_PAD_VALUE, 0, 0, NULL_PAD_VALUE]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=1, end=3),
                                   np.array([NULL_PAD_VALUE, 0, 0, 1]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=2, end=3),
                                   np.array([NULL_PAD_VALUE, NULL_PAD_VALUE, 0, 1]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=2, end=2),
                                   np.array([NULL_PAD_VALUE, NULL_PAD_VALUE, 0, NULL_PAD_VALUE]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=2, end=None),
                                   np.array([NULL_PAD_VALUE, NULL_PAD_VALUE, 0, 1]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=None, end=None),
                                   np.array([1, 0, 0, 1]))
        np.testing.assert_allclose(null_pad(np.array([1, 0, 0, 1]), start=None, end=0),
                                   np.array([1, NULL_PAD_VALUE, NULL_PAD_VALUE, NULL_PAD_VALUE]))

    def test_add_remove_predicted(self):
        k = "test"
        out_k = remove_predicted(add_predicted(k))
        self.assertEqual(k, out_k)

    def test_add_remove_predicted_dict(self):
        d = {
            add_predicted("test1"): 1,
            "test2": 2,
        }
        expected_d = {
            "test1": 1,
            "test2": 2,
        }
        out_d = remove_predicted_from_dict(d)
        self.assertEqual(expected_d, out_d)


if __name__ == '__main__':
    unittest.main()
