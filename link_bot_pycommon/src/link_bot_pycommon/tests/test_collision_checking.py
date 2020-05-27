from unittest import TestCase

import numpy as np

from link_bot_pycommon.collision_checking import batch_in_collision_tf, inflate_tf, gripper_interpolate_cc_and_oob
from moonshine.tests.testing_utils import assert_close_tf


class Test(TestCase):
    def test_inflate_tf1(self):
        env = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0]
        ], dtype=np.float32)
        expected_inflated_env = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ], dtype=np.float32)
        inflated_env = inflate_tf(env, radius_m=0.01, res=0.01)
        assert_close_tf(inflated_env, expected_inflated_env)

    def test_inflate_tf2(self):
        env = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ], dtype=np.float32)
        expected_inflated_env = np.array([
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ], dtype=np.float32)
        inflated_env = inflate_tf(env, radius_m=0.02, res=0.01)
        assert_close_tf(inflated_env, expected_inflated_env)

    def test_gripper_collision(self):
        env = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ], dtype=np.float32)
        e = {
            'full_env/env': env,
            'full_env/res': 1,
            'full_env/origin': [0, 0],
        }
        self.assertTrue(batch_in_collision_tf(environment=e, xs=np.array([0]), ys=np.array([0]), inflate_radius_m=1).numpy())
        self.assertTrue(batch_in_collision_tf(environment=e, xs=np.array([1]), ys=np.array([0]), inflate_radius_m=1).numpy())
        self.assertFalse(batch_in_collision_tf(environment=e, xs=np.array([2]), ys=np.array([0]), inflate_radius_m=1).numpy())
        self.assertFalse(batch_in_collision_tf(environment=e, xs=np.array([3]), ys=np.array([0]), inflate_radius_m=1).numpy())
        self.assertFalse(batch_in_collision_tf(environment=e, xs=np.array([0]), ys=np.array([2]), inflate_radius_m=1).numpy())
        self.assertTrue(batch_in_collision_tf(environment=e, xs=np.array([3]), ys=np.array([2]), inflate_radius_m=1).numpy())

    def test_gripper_interpolate_cc_and_oob(self):
        env = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0]
            # When inflated:
            # [1, 1, 0, 0],
            # [1, 1, 0, 0],
            # [0, 1, 1, 1],
            # [0, 1, 1, 1]
        ], dtype=np.float32)
        e = {
            'full_env/env': env,
            'full_env/res': 1,
            'full_env/origin': [0, 0],
        }
        self.assertTrue(gripper_interpolate_cc_and_oob(e, np.array([0, 0]), np.array([1, 0]), inflate_radius_m=1).numpy())
        self.assertTrue(gripper_interpolate_cc_and_oob(e, np.array([0, 0]), np.array([3, 0]), inflate_radius_m=1).numpy())
        self.assertTrue(gripper_interpolate_cc_and_oob(e, np.array([1, 1]), np.array([0, 2]), inflate_radius_m=1).numpy())
        self.assertTrue(gripper_interpolate_cc_and_oob(e, np.array([0, 3]), np.array([3, 3]), inflate_radius_m=1).numpy())
        self.assertFalse(gripper_interpolate_cc_and_oob(e, np.array([2, 0]), np.array([3, 0]), inflate_radius_m=1).numpy())
        self.assertFalse(gripper_interpolate_cc_and_oob(e, np.array([2, 0]), np.array([3, 1]), inflate_radius_m=1).numpy())
        self.assertFalse(gripper_interpolate_cc_and_oob(e, np.array([0, 2]), np.array([0, 3]), inflate_radius_m=1).numpy())
