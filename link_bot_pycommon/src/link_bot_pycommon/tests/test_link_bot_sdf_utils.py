from unittest import TestCase

import numpy as np

from link_bot_pycommon.link_bot_sdf_utils import compute_extent_3d


class Test(TestCase):
    def test_compute_extent_3d(self):
        actual = compute_extent_3d(rows=200, cols=200, channels=1, resolution=0.01)
        desired = np.array([-1.0, 1.0, -1.0, 1.0, 0.0, 0.01])
        np.testing.assert_allclose(actual, desired)

    def test_compute_extent_3d_2(self):
        actual = compute_extent_3d(rows=200, cols=200, channels=200, resolution=0.01)
        desired = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        np.testing.assert_allclose(actual, desired)
