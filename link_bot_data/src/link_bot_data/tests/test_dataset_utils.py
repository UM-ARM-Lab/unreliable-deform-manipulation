import unittest
import tensorflow as tf

from link_bot_data.link_bot_dataset_utils import is_funneling


class MyTestCase(unittest.TestCase):
    def test_is_funneling(self):
        self.assertEqual(is_funneling(tf.constant([1, 0, 0, 1], tf.int64)).numpy(), True)
        self.assertEqual(is_funneling(tf.constant([1, 0, 0, 0], tf.int64)).numpy(), False)

if __name__ == '__main__':
    unittest.main()
