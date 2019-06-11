import tensorflow as tf


def ravel_2d(indeces_batch, cols):
    return indeces_batch[:, :, 0] * cols + indeces_batch[:, :, 1]


@tf.custom_gradient
def sdf_func(sdf_batch, gradient_batch, resolution_batch, origin_batch, coordinates_batch, constraint_space_dim):
    float_coordinates = tf.math.divide(coordinates_batch, resolution_batch)
    integer_coordinates = tf.cast(float_coordinates, dtype=tf.int32)
    integer_coordinates = tf.add(integer_coordinates, origin_batch, name='integer_coordinates')
    integer_coordinates = tf.reshape(integer_coordinates, [-1, constraint_space_dim])

    oob_left = integer_coordinates[:, 0] < 0
    oob_right = integer_coordinates[:, 0] >= sdf_batch.shape[2]
    oob_up = integer_coordinates[:, 1] < 0
    oob_down = integer_coordinates[:, 1] >= sdf_batch.shape[1]
    out_of_bounds = tf.math.reduce_any(tf.stack((oob_up, oob_down, oob_left, oob_right), axis=1), axis=1)
    integer_coordinates = tf.expand_dims(integer_coordinates, axis=1)
    integer_coordinates_ravel = ravel_2d(integer_coordinates, sdf_batch.shape[2])
    sdf_batch_flat = tf.reshape(sdf_batch, [-1, sdf_batch.shape[1] * sdf_batch.shape[2]])
    gradient_batch_flat = tf.reshape(gradient_batch,
                                     [-1, gradient_batch.shape[1] * gradient_batch.shape[2], gradient_batch.shape[3]])

    sdf_value = tf.batch_gather(sdf_batch_flat, integer_coordinates_ravel, name='sdf_gather')
    sdf_value = tf.reshape(sdf_value, [-1, 1], name='index_sdfs')

    # TODO: how to handle out of bounds??? current set to slightly inside an obstacle
    oob_value = tf.ones_like(sdf_value) * -0.1
    sdf_value = tf.where(out_of_bounds, oob_value, sdf_value, name='sdf_value')

    def __sdf_gradient_func(dy):
        sdf_gradient = tf.batch_gather(gradient_batch_flat, integer_coordinates_ravel, name='sdf_gradients_gather')
        sdf_gradient = tf.reshape(sdf_gradient, [-1, constraint_space_dim], name='index_sdf_gradient')
        # TODO: how to handle out of bounds??? current just zero everything
        # oob_gradient = tf.ones_like(sdf_gradient) * 0.0
        # sdf_gradient = tf.where(out_of_bounds, oob_gradient, sdf_gradient, name='sdf_gradient')
        return None, None, None, None, dy * sdf_gradient, None

    return sdf_value, __sdf_gradient_func
