import numpy as np
import tensorflow as tf


@tf.custom_gradient
def sdf_func(sdf, full_sdf_gradient, sdf_resolution, sdf_origin_coordinate, sdf_coordinates, P):
    float_coordinates = tf.math.divide(sdf_coordinates, sdf_resolution)
    integer_coordinates = tf.cast(float_coordinates, dtype=tf.int32)
    integer_coordinates = tf.reshape(integer_coordinates, [-1, P])
    integer_coordinates = tf.add(integer_coordinates, sdf_origin_coordinate, name='integer_coordinates')
    # blindly assume the point is within our grid

    oob_left = integer_coordinates[:, 0] < 0
    oob_right = integer_coordinates[:, 0] >= sdf.shape[1]
    oob_up = integer_coordinates[:, 1] < 0
    oob_down = integer_coordinates[:, 1] >= sdf.shape[0]
    out_of_bounds = tf.math.reduce_any(tf.stack((oob_up, oob_down, oob_left, oob_right), axis=1), axis=1)

    sdf_value = tf.gather_nd(sdf, integer_coordinates, name='sdf_gather')
    sdf_value = tf.reshape(sdf_value, [-1, 1], name='index_sdfs')

    # TODO: how to handle out of bounds???
    oob_value = tf.ones_like(sdf_value) * 0.0

    sdf_value = tf.where(out_of_bounds, oob_value, sdf_value, name='sdf_value')

    def __sdf_gradient_func(dy):
        sdf_gradient = tf.gather_nd(full_sdf_gradient, integer_coordinates, name='sdf_gradients_gather')
        sdf_gradient = tf.reshape(sdf_gradient, [-1, P], name='index_sdf_gradient')
        # TODO: how to handle out of bounds???
        oob_gradient = tf.ones_like(sdf_gradient) * 0.0
        sdf_gradient = tf.where(out_of_bounds, oob_gradient, sdf_gradient, name='sdf_gradient')
        return None, None, None, None, dy * sdf_gradient, None

    return sdf_value, __sdf_gradient_func
