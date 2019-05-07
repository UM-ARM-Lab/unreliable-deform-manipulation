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
        batch_size = 2
        n_steps = 2

        args = {
            'debug': False,
            'log': False,
            'checkpoint': None,
            'seed': 999,
            'random_init': False,
        }

        N = 3
        M = 2
        L = 2
        P = 2
        Q = 1

        cls.goal = np.array([[2, 1, 3]])

        W = 10
        H = 20
        sdf = np.ones((H, W), dtype=np.float32)
        sdf[H // 2 + 1, W // 2 + 3] = -1
        sdf_gradient = np.zeros((W, H, 2), dtype=np.float32)
        sdf_resolution = np.array([1, 1], dtype=np.float32)

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
            tf.assign(cls.model.R_c, R_d_init),
            tf.assign(cls.model.A_c, A_d_init),
            tf.assign(cls.model.B_c, B_d_init),
            tf.assign(cls.model.R_k, R_k_init),
            tf.assign(cls.model.A_k, A_k_init),
            tf.assign(cls.model.B_k, B_k_init),
            tf.assign(cls.model.threshold_k, k_threshold_init),
        ]
        cls.model.sess.run(assign_weights_ops)

        # Construct inputs
        s = np.array([
            [[3, 7, 4],
             [1, 2, 3],
             [2, -1, -3]],
            [[-1, 2, -1],
             [1, 1, 2],
             [1, 2, -2]],
        ])

        u = np.array([
            [[0, 1],
             [1, 1]],
            [[1, 0],
             [1, 0]],
        ])

        cls.c = np.linalg.norm(s - cls.goal, axis=1)
        cls.k = np.array([
            [[0],
             [1],
             [0]],
            [[0],
             [0],
             [0]],
        ])

        k_mask_indeces_2d = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 2],
        ])

        cls.feed_dict = {
            cls.model.observations: s,
            cls.model.u: u,
            cls.model.observation_goal: cls.goal,
            cls.model.c_label: cls.c,
            cls.model.k_label: cls.k,
            cls.model.k_mask_indeces_2d: k_mask_indeces_2d
        }

    def test_hat_latent_c(self):
        expected_hat_latent_c = np.array([
            [[3, 7],
             [1, 2],
             [2, -1]],
            [[-1, 2],
             [1, 1],
             [1, 2]],
        ])

        hat_latent_c = self.model.sess.run(self.model.hat_latent_c, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_latent_c, expected_hat_latent_c)

    def test_hat_latent_c_next(self):
        expected_hat_latent_c_next = np.array([
            [[3, 7],
             [3, 7.1],
             [3.1, 7.2]],
            [[-1, 2],
             [-0.9, 2],
             [-0.8, 2]],
        ])

        hat_latent_c_next = self.model.sess.run(self.model.hat_latent_c_next, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_latent_c_next, expected_hat_latent_c_next)

    def test_hat_o_k(self):
        expected_hat_o_k = np.array([
            [[3, 4],
             [1, 3],
             [2, -3]],
            [[-1, -1],
             [1, 2],
             [1, -2]],
        ])

        hat_o_k = self.model.sess.run(self.model.hat_latent_k, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_o_k, expected_hat_o_k)

    def test_hat_o_k_next(self):
        expected_hat_o_k_next = np.array([
            [[3, 4],
             [3, 4.1],
             [3.1, 4.2]],
            [[-1, -1],
             [-0.9, -1],
             [-0.8, -1]],
        ])

        hat_o_k_next = self.model.sess.run(self.model.hat_latent_k_next, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_o_k_next, expected_hat_o_k_next)

    def test_error_to_goal(self):
        expected_d_to_goal = np.array([
            [[2 - 3, 1 - 7],
             [2 - 1, 1 - 2],
             [2 - 2, 1 - -1]],
            [[2 - -1, 1 - 2],
             [2 - 1, 1 - 1],
             [2 - 1, 1 - 2]],
        ])

        hat_d_to_goal = self.model.sess.run(self.model.d_to_goal, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_d_to_goal, expected_d_to_goal)

    def test_cost_loss(self):
        expected_c = np.linalg.norm(np.array([
            [[2 - 3, 1 - 7],
             [2 - 1, 1 - 2],
             [2 - 2, 1 - -1]],
            [[2 - -1, 1 - 2],
             [2 - 1, 1 - 1],
             [2 - 1, 1 - 2]],
        ]), axis=2) ** 2

        expected_c_error = (expected_c - self.c) ** 2
        expected_c_error_masked = np.array([
            expected_c_error[0, 0],  # only the first time step should be un-masked
            expected_c_error[1, 0],  # for the second trajectory, all steps are un-masked
            expected_c_error[1, 1],
            expected_c_error[1, 2],
        ])
        expected_c_loss = np.mean(expected_c_error_masked)

        hat_c = self.model.sess.run(self.model.hat_c, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_c, expected_c)

        hat_c_error = self.model.sess.run(self.model.all_cost_prediction_error, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_c_error, expected_c_error, rtol=2e-5)

        hat_c_error_masked = self.model.sess.run(self.model.cost_prediction_error, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_c_error_masked, expected_c_error_masked, rtol=2e-5)

        c_loss = self.model.sess.run(self.model.cost_prediction_loss, feed_dict=self.feed_dict)
        np.testing.assert_allclose(c_loss, expected_c_loss)

    def test_dynamics_loss(self):
        expected_state_prediction_error_in_c = np.array([
            [0,
             (3 - 1) ** 2 + (7.1 - 2) ** 2,
             (3.1 - 2) ** 2 + (7.2 - -1) ** 2],
            [0,
             (-0.9 - 1) ** 2 + (2 - 1) ** 2,
             (-0.8 - 1) ** 2 + (2 - 2) ** 2],
        ])

        expected_state_prediction_error_in_c_masked = np.array([
            0,
            0,
            (-0.9 - 1) ** 2 + (2 - 1) ** 2,
            (-0.8 - 1) ** 2 + (2 - 2) ** 2,
        ])

        expected_sd_loss = np.mean(expected_state_prediction_error_in_c_masked)

        state_prediction_error_in_c = self.model.sess.run(self.model.all_state_prediction_error_in_c,
                                                          feed_dict=self.feed_dict)
        np.testing.assert_allclose(state_prediction_error_in_c, expected_state_prediction_error_in_c, rtol=2e-5)

        state_prediction_error_in_c_masked = self.model.sess.run(self.model.state_prediction_error_in_c,
                                                                 feed_dict=self.feed_dict)
        np.testing.assert_allclose(state_prediction_error_in_c_masked, expected_state_prediction_error_in_c_masked,
                                   rtol=2e-5)

        sd_loss = self.model.sess.run(self.model.state_prediction_loss_in_c, feed_dict=self.feed_dict)
        np.testing.assert_allclose(sd_loss, expected_sd_loss)

    def test_constraint_dynamics_loss(self):
        expected_state_prediction_error_in_k = np.array([
            [0,
             (3 - 1) ** 2 + (4.1 - 3) ** 2,
             (3.1 - 2) ** 2 + (4.2 - -3) ** 2],
            [0,
             (-0.9 - 1) ** 2 + (-1.0 - 2) ** 2,
             (-0.8 - 1) ** 2 + (-1.0 - -2) ** 2],
        ])

        expected_state_prediction_error_in_k_masked = np.array([
            0,
            0,
            (-0.9 - 1) ** 2 + (-1.0 - 2) ** 2,
            (-0.8 - 1) ** 2 + (-1.0 - -2) ** 2,
        ])

        expected_sd_loss = np.mean(expected_state_prediction_error_in_k_masked)

        state_prediction_error_in_k = self.model.sess.run(self.model.all_state_prediction_error_in_k,
                                                          feed_dict=self.feed_dict)
        np.testing.assert_allclose(state_prediction_error_in_k, expected_state_prediction_error_in_k, rtol=2e-5)

        state_prediction_error_in_k_masked = self.model.sess.run(self.model.state_prediction_error_in_k,
                                                                 feed_dict=self.feed_dict)
        np.testing.assert_allclose(state_prediction_error_in_k_masked, expected_state_prediction_error_in_k_masked,
                                   rtol=2e-5)

        sd_loss = self.model.sess.run(self.model.state_prediction_loss_in_k, feed_dict=self.feed_dict)
        np.testing.assert_allclose(sd_loss, expected_sd_loss)

    def test_constraint_loss(self):
        expected_k_violated = np.array([
            [[False],
             [True],
             [False]],
            [[False],
             [False],
             [False]],
        ])

        expected_k = np.array([
            [[-0.5],
             [1.5],
             [-0.5]],
            [[-0.5],
             [-0.5],
             [-0.5]],
        ]) * 100

        expected_k_loss = np.array([0], dtype=np.float32)

        hat_k_violated = self.model.sess.run(self.model.hat_k_violated, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_k_violated, expected_k_violated)

        hat_k = self.model.sess.run(self.model.hat_k, feed_dict=self.feed_dict)
        np.testing.assert_allclose(hat_k, expected_k)

        k_loss = self.model.sess.run(self.model.constraint_prediction_loss, feed_dict=self.feed_dict)
        np.testing.assert_allclose(k_loss, expected_k_loss, atol=1e-20)


if __name__ == '__main__':
    unittest.main()
