# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, Mark van der Wilk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import gpflow as gpf
import tensorflow as tf


class MySVGP(gpf.models.SVGP):
    """
    Testing adding additional terms to the SVGP loss
    """

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

        # TODO: compute this from the input data? or at least make it an argument to the constructor
        nominal_length = 0.23
        beta = 1000
        # if the number is high, then we say the likelihood is low
        tail_to_mid = tf.norm(predicted_points[:, 0] - predicted_points[:, 1], axis=1)
        mid_to_head = tf.norm(predicted_points[:, 1] - predicted_points[:, 2], axis=1)
        length_penalty = beta * tf.reduce_mean(tf.abs(tail_to_mid - nominal_length) + tf.abs(mid_to_head - nominal_length))

        # x_points = tf.reshape(x_points_flat, [self.num_data, 3, 2])
        # likelihood = tf.Print(likelihood, data=[x_points[:, 0]], summarize=100, message='tail')
        # likelihood = tf.Print(likelihood, data=[x_points[:, 1]], summarize=100, message='mid')
        # likelihood = tf.Print(likelihood, data=[x_points[:, 2]], summarize=100, message='head')
        # likelihood = tf.Print(likelihood, data=[tail_to_mid], summarize=100, message='tail to mid')

        likelihood = 1 * likelihood
        my_likelihood = likelihood - length_penalty

        my_likelihood = tf.Print(my_likelihood, data=[likelihood, length_penalty, my_likelihood], message='loss terms: ')

        # return likelihood
        return my_likelihood
