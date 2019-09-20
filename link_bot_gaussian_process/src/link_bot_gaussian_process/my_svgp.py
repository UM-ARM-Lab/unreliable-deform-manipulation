import gpflow as gpf
import numpy as np
import tensorflow as tf


class MySVGP(gpf.models.SVGP):
    """
    Testing adding additional terms to the SVGP loss
    """

    def __init__(self, X, Y, kern, likelihood, beta, feat=None, **kwargs):
        super().__init__(X, Y, kern, likelihood, feat, **kwargs)

        self.beta = beta
        head_zeros = np.zeros(shape=(self.num_data, 2))
        x_points_flat = np.concatenate((X[:, 0:4], head_zeros), axis=1)
        x_points = np.reshape(x_points_flat, [self.num_data, -1, 2])
        tail_to_mid = np.linalg.norm(x_points[:, 0] - x_points[:, 1], axis=1)
        mid_to_head = np.linalg.norm(x_points[:, 1] - x_points[:, 2], axis=1)
        distances = np.concatenate((tail_to_mid, mid_to_head), axis=0)
        self.nominal_length = np.mean(distances)

        # Just a sanity check since our Gazebo rope is rigid at the moment
        assert np.std(distances) < 1e-5

    @gpf.params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """
        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict(self.X, full_cov=False, full_output_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, gpf.settings.float_type) / tf.cast(tf.shape(self.X)[0], gpf.settings.float_type)

        likelihood = tf.reduce_sum(var_exp) * scale - KL

        # fmean represents the deltas of each coordinate:
        # [dx1, dy1, dx2, dy2, dx3, dy3]
        # first construct the new rope configuration
        # the first four points of X are
        # [x1, y1, x2, y2], and (x3,y3) is not included because it's always (0, 0)
        head_zeros = tf.zeros(shape=(self.num_data, 2))
        x_points_flat = tf.concat((self.X[:, 0:4], head_zeros), axis=1)
        predicted_points = tf.reshape(x_points_flat + fmean, [self.num_data, -1, 2])
        # now each row in points looks like [x1, y1, x2, y2, x3, y3] where 1 is tail and 3 is head
        # so we penalize abs(dist(1,2) - nominal_length) + abs(dist(2,3) - nominal_length)

        # if the number is high, then we say the likelihood is low
        tail_to_mid = tf.norm(predicted_points[:, 0] - predicted_points[:, 1], axis=1)
        mid_to_head = tf.norm(predicted_points[:, 1] - predicted_points[:, 2], axis=1)
        length_penalty = self.beta * tf.reduce_mean(
            tf.abs(tail_to_mid - self.nominal_length) + tf.abs(mid_to_head - self.nominal_length))

        my_likelihood = likelihood - length_penalty

        # return likelihood
        return my_likelihood
