from unittest import TestCase

import numpy as np

from link_bot_pycommon.link_bot_sdf_utils import compute_extent_3d, extent_to_env_size


class Test(TestCase):
    def test_compute_extent_3d(self):
        actual = compute_extent_3d(rows=200, cols=200, channels=1, resolution=0.01)
        desired = np.array([-1.0, 1.0, -1.0, 1.0, 0.0, 0.01])
        np.testing.assert_allclose(actual, desired)

    def test_compute_extent_3d_2(self):
        actual = compute_extent_3d(rows=200, cols=200, channels=200, resolution=0.01)
        desired = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        np.testing.assert_allclose(actual, desired)

    def test_compute_extent_3d_3(self):
        actual = compute_extent_3d(rows=100, cols=200, channels=1, resolution=0.01)
        desired = np.array([-1.0, 1.0, -0.5, 0.5, 0.0, 0.01])
        np.testing.assert_allclose(actual, desired)

    def test_extent_to_env_size(self):
        extent = [-1, 1, -0.5, 0.5, 0, 0.5]
        env_h_m, env_w_m, env_c_m = extent_to_env_size(extent)
        self.assertEqual(env_h_m, 1)
        self.assertEqual(env_w_m, 2)
        self.assertEqual(env_c_m, 0.5)
