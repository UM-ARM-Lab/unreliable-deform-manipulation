import unittest

import gpflow as gpf
import numpy as np

from link_bot_gaussian_process.moment_matching import predict


class MomentMatchingTest(unittest.TestCase):

    def test_univariate(self):
        X = np.array([-0.8, -0.5, -0.15, -0.12, 0.4, 0.7]).reshape(-1, 1)
        Y = np.array([0.3, 0.0, 0.25, 0.27, -0.12, 0.0]).reshape(-1, 1)
        N, _ = X.shape

        kernel = gpf.kernels.SquaredExponential(1, lengthscales=0.1)
        m = gpf.models.GPR(X, Y, kernel)
        models = [m]
        opt = gpf.train.ScipyOptimizer()
        opt.minimize(m)

        mu_tilde_t = np.array([0.3])
        sigma_tilde_t = np.array([[0.08]])
        prior = (mu_tilde_t, sigma_tilde_t)

        posterior = predict(models, X, Y, prior)
        mu_tilde_t_plus_1, sigma_tilde_t_plus_1 = posterior
        self.assertAlmostEqual(mu_tilde_t_plus_1[0, 0], 0.0205628, places=6)
        self.assertAlmostEqual(sigma_tilde_t_plus_1[0, 0], 0.023229, places=6)

    def test_control_input(self):
        X = np.array([[-0.8, -0.8], [-0.5, -0.5], [-0.15, -0.15], [-0.12, -0.12], [0.4, 0.4], [0.7, 0.7]])
        Y = np.array([0.3, 0.0, 0.25, 0.27, -0.12, 0.0]).reshape(-1, 1)
        N, _ = X.shape

        kernel = gpf.kernels.SquaredExponential(2, lengthscales=[0.1, 0.1])
        m = gpf.models.GPR(X, Y, kernel)
        models = [m]
        opt = gpf.train.ScipyOptimizer()
        opt.minimize(m)
        print(m)

        mu_tilde_t = np.array([0.3, 1.0])
        # we would need covariance between state and control here. How do we estimate that?
        sigma_tilde_t = np.array([[0.08, 0.1], [0.1, 1.0]])
        prior = (mu_tilde_t, sigma_tilde_t)

        posterior = predict(models, X, Y, prior)
        mu_tilde_t_plus_1, sigma_tilde_t_plus_1 = posterior
        self.assertAlmostEqual(mu_tilde_t_plus_1[0, 0], 0.006905, places=6)
        self.assertAlmostEqual(sigma_tilde_t_plus_1[0, 0], 0.028302, places=6)
