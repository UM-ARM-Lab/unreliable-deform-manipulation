#!/usr/bin/env python
from __future__ import print_function

import unittest
import numpy as np

from link_bot_notebooks import toy_problem_optimization_common as tpo


class TestLoadTrain(unittest.TestCase):

    def test_shape(self):
        trajectory_length_during_collection = 4
        trajectory_length_to_train = 1
        n_trajs_collected = 2
        d = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
                      [10, 11, 12], [11, 12, 13], [12, 13, 14], [13, 14, 15], [14, 15, 16]])
        x = tpo.load_train2(d, [0, 1], trajectory_length_during_collection, trajectory_length_to_train)
        self.assertEqual(x.shape[0], trajectory_length_to_train + 1)
        self.assertEqual(x.shape[1], 2)
        self.assertEqual(x.shape[2],
                         (trajectory_length_during_collection - trajectory_length_to_train) * n_trajs_collected)
        self.assertEqual(x[0, 0, 0], 0)
        self.assertEqual(x[1, 0, 0], 1)
        self.assertEqual(x[0, 0, 1], 1)
        self.assertEqual(x[1, 0, 1], 2)
        self.assertEqual(x[0, 0, 2], 2)
        self.assertEqual(x[1, 0, 2], 3)

        self.assertEqual(x[0, 1, 0], 1)
        self.assertEqual(x[1, 1, 0], 2)


if __name__ == '__main__':
    unittest.main()
