#!/usr/bin/env python
from __future__ import print_function

import unittest
import numpy as np

from link_bot_notebooks import notebook_finder


class TestExtractor(unittest.TestCase):

    def test_two_link_pos_vel(self):
        from link_bot_notebooks import toy_problem_optimization_common as tpo

        g = np.zeros((6, 1))
        d = np.array([1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 7, 8])
        s, u, c, = tpo.two_link_pos_vel_extractor(d, g)
        np.testing.assert_allclose(s, np.array([[1], [2], [3], [4], [5], [6]]))
        np.testing.assert_allclose(u, np.array([[7], [8]]))
        np.testing.assert_allclose(c, np.array([1 * 1 + 2 * 2]))


    def test_link_pos_vel(self):
        from link_bot_notebooks import toy_problem_optimization_common as tpo

        g = np.zeros((6, 1))
        d = np.array([1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 7, 8])
        s, u, c, = tpo.link_pos_vel_extractor(6)(d, g)
        np.testing.assert_allclose(s, np.array([[1], [2], [3], [4], [5], [6]]))
        np.testing.assert_allclose(u, np.array([[7], [8]]))
        np.testing.assert_allclose(c, np.array([1 * 1 + 2 * 2]))


if __name__ == '__main__':
    unittest.main()
