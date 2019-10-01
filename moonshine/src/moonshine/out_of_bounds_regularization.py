import numpy as np
import tensorflow as tf


class OutOfBoundsRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, sdf_extent, beta):
        self.sdf_extent = sdf_extent
        self.beta = beta

    def __call__(self, sdf_input_points):
        # FIXME: this assumes that the physical world coordinates (0,0) in meters is the origin/center of the SDF
        distances_to_origin = tf.norm(sdf_input_points, axis=1)
        oob_left = sdf_input_points[:, 0] <= self.sdf_extent[0]
        oob_right = sdf_input_points[:, 0] >= self.sdf_extent[1]
        oob_up = sdf_input_points[:, 1] <= self.sdf_extent[2]
        oob_down = sdf_input_points[:, 1] >= self.sdf_extent[3]
        out_of_bounds = tf.math.reduce_any(tf.stack((oob_up, oob_down, oob_left, oob_right), axis=1), axis=1)
        in_bounds_value = tf.ones_like(distances_to_origin) * 0.0
        distances_out_of_bounds = tf.where(out_of_bounds, distances_to_origin, in_bounds_value)
        out_of_bounds_loss = tf.reduce_mean(distances_out_of_bounds)
        return tf.norm(out_of_bounds_loss) * self.beta

    def get_config(self):
        config = {
            'sdf_extent': self.sdf_extent,
            'beta': self.beta
        }
        return config
