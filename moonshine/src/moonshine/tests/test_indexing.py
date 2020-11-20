import unittest

import tensorflow as tf

from moonshine.indexing import index_dict_of_batched_tensors_tf
from moonshine.tests import testing_utils


class TestIndexing(unittest.TestCase):

    def test_index_dict_of_batched_tensors_tf(self):
        input_dict = {
            'a': tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=tf.float32),
            'b': tf.constant([[0, 2], [3, 5], [6, 8]], dtype=tf.float32),
        }

        expected_dict = {
            'a': tf.constant([0, 1, 2], dtype=tf.float32),
            'b': tf.constant([0, 2], dtype=tf.float32),
        }
        out_dict = index_dict_of_batched_tensors_tf(input_dict, index=0, batch_axis=0, keep_dims=False)
        testing_utils.assert_dicts_close_tf(out_dict, expected_dict)

        expected_dict = {
            'a': tf.constant([[0, 1, 2]], dtype=tf.float32),
            'b': tf.constant([[0, 2]], dtype=tf.float32),
        }
        out_dict = index_dict_of_batched_tensors_tf(input_dict, index=0, batch_axis=0, keep_dims=True)
        testing_utils.assert_dicts_close_tf(out_dict, expected_dict)

        expected_dict = {
            'a': tf.constant([0, 3, 6], dtype=tf.float32),
            'b': tf.constant([0, 3, 6], dtype=tf.float32),
        }
        out_dict = index_dict_of_batched_tensors_tf(input_dict, index=0, batch_axis=1, keep_dims=False)
        testing_utils.assert_dicts_close_tf(out_dict, expected_dict)

        expected_dict = {
            'a': tf.constant([[0], [3], [6]], dtype=tf.float32),
            'b': tf.constant([[0], [3], [6]], dtype=tf.float32),
        }
        out_dict = index_dict_of_batched_tensors_tf(input_dict, index=0, batch_axis=1, keep_dims=True)
        testing_utils.assert_dicts_close_tf(out_dict, expected_dict)


if __name__ == '__main__':
    unittest.main()
