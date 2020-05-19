import unittest
import numpy as np
import tensorflow as tf

from link_bot_data.link_bot_dataset_utils import is_reconverging, null_pad, NULL_PAD_VALUE


class MyTestCase(unittest.TestCase):
    def test_is_reconverging(self):
        self.assertTrue(is_reconverging(tf.constant([1, 0, 0, 1], tf.int64)).numpy())
        self.assertFalse(is_reconverging(tf.constant([1, 0, 0, 0], tf.int64)).numpy())

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


if __name__ == '__main__':
    unittest.main()
