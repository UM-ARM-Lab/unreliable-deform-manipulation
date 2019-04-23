#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import unittest
import numpy as np

from link_bot_notebooks import linear_constraint_model as m


class TestLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dt = 0.1
        batch_size = 1
        n_steps = 2

        args = {
            'debug': False,
            'log': False,
            'checkpoint': None,
            'seed': 999,
        }

        N = 3
        M = 2
        L = 2
        P = 2
        Q = 1

        cls.goal = np.array([[2, 1, 3]])

        W = 10
        H = 20
        sdf = np.random.randn(W, H).astype(np.float32)
        sdf_gradient = np.random.randn(W, H, 2).astype(np.float32)
        sdf_resolution = np.random.randn(2).astype(np.float32)

        cls.model = m.LinearConstraintModel(args, sdf, sdf_gradient, sdf_resolution, batch_size, N, M, L, P, Q, dt,
                                            n_steps)
        cls.model.setup()

        # Manually set weights
        R_d_init = np.zeros((N, M), dtype=np.float32)
        R_d_init[0, 0] = 1
        R_d_init[1, 1] = 1
        A_d_init = np.zeros((M, M), dtype=np.float32)
        B_d_init = np.zeros((M, L), dtype=np.float32)
        np.fill_diagonal(B_d_init, 1)
        R_k_init = np.zeros((N, P), dtype=np.float32)
        R_k_init[0, 0] = 1.0
        R_k_init[2, 1] = 1.0
        A_k_init = np.zeros((P, P), dtype=np.float32)
        B_k_init = np.zeros((P, L), dtype=np.float32)
        np.fill_diagonal(B_k_init, 1)
        k_threshold_init = 0.5

        assign_weights_ops = [
            tf.assign(cls.model.R_d, R_d_init),
            tf.assign(cls.model.A_d, A_d_init),
            tf.assign(cls.model.B_d, B_d_init),
            tf.assign(cls.model.R_k, R_k_init),
            tf.assign(cls.model.A_k, A_k_init),
            tf.assign(cls.model.B_k, B_k_init),
            tf.assign(cls.model.threshold_k, k_threshold_init),
        ]
        cls.model.sess.run(assign_weights_ops)

        # Construct inputs
        s = np.array([
            [3, 7, 4],
            [1, 2, 3],
            [2, -1, -3],
        ])

        u = np.array([
            [0, 1],
            [1, 1],
        ])

        cls.c = np.linalg.norm(s - cls.goal, axis=1)
        cls.k = np.array([
            [0],
            [0],
            [1],
        ])

        cls.feed_dict = {
            cls.model.s: np.expand_dims(s, axis=0),
            cls.model.u: np.expand_dims(u, axis=0),
            cls.model.s_goal: cls.goal,
            cls.model.c_label: np.expand_dims(cls.c, axis=0),
            cls.model.k_label: np.expand_dims(cls.k, axis=0)
        }

    def test_hat_o_d(self):
        # expected outputs
        expected_hat_o_d = np.array([[
            [3, 7],
            [1, 2],
            [2, -1],
        ]])

        # compute outputs and test for the right output
        hat_o_d = self.model.sess.run(self.model.hat_o_d, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_o_d, expected_hat_o_d)

    def test_hat_o_d_next(self):
        # expected outputs
        expected_hat_o_d_next = np.array([[
            [3, 7],
            [3, 7.1],
            [3.1, 7.2],
        ]])

        # compute outputs and test for the right output
        hat_o_d_next = self.model.sess.run(self.model.hat_o_d_next, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_o_d_next, expected_hat_o_d_next)

    def test_hat_o_k(self):
        # expected outputs
        expected_hat_o_k = np.array([[
            [3, 4],
            [1, 3],
            [2, -3],
        ]])

        # compute outputs and test for the right output
        hat_o_k = self.model.sess.run(self.model.hat_o_k, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_o_k, expected_hat_o_k)

    def test_hat_o_k_next(self):
        # expected outputs
        expected_hat_o_k_next = np.array([[
            [3, 4],
            [3, 4.1],
            [3.1, 4.2],
        ]])

        # compute outputs and test for the right output
        hat_o_k_next = self.model.sess.run(self.model.hat_o_k_next, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_o_k_next, expected_hat_o_k_next)

    def test_error_to_goal(self):
        # expected outputs
        expected_d_to_goal = np.array([[
            [2 - 3, 1 - 7],
            [2 - 1, 1 - 2],
            [2 - 2, 1 - -1],
        ]])

        # compute outputs and test for the right output
        hat_d_to_goal = self.model.sess.run(self.model.d_to_goal, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_d_to_goal, expected_d_to_goal)

    def test_cost_loss(self):
        # expected outputs
        expected_c = np.linalg.norm(np.array([[
            [2 - 3, 1 - 7],
            [2 - 1, 1 - 2],
            [2 - 2, 1 - -1],
        ]]), axis=2) ** 2

        expected_c_error = (expected_c - self.c) ** 2
        expected_c_error_masked = np.array([
            expected_c_error[0, 0],
            expected_c_error[0, 1]
        ])
        expected_c_loss = np.mean(expected_c_error_masked)

        # compute outputs and test for the right output
        hat_c = self.model.sess.run(self.model.hat_c, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_c, expected_c)

        hat_c_error = self.model.sess.run(self.model.all_cost_prediction_error, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_c_error, expected_c_error, rtol=2e-5)

        hat_c_error_masked = self.model.sess.run(self.model.cost_prediction_error, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_c_error_masked, expected_c_error_masked, rtol=2e-5)

        c_loss = self.model.sess.run(self.model.cost_prediction_loss, feed_dict=self.feed_dict)
        np.testing.assert_allclose(c_loss, expected_c_loss)


if __name__ == '__main__':
    unittest.main()
