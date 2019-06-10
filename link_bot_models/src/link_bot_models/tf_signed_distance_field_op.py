import tensorflow as tf


@tf.custom_gradient
def sdf_func(sdf, full_sdf_gradient, sdf_resolution, sdf_origin_coordinate, sdf_coordinates, P):
    float_coordinates = tf.math.divide(sdf_coordinates, sdf_resolution)
    integer_coordinates = tf.cast(float_coordinates, dtype=tf.int32)
    integer_coordinates = tf.reshape(integer_coordinates, [-1, P])
    integer_coordinates = tf.add(integer_coordinates, sdf_origin_coordinate, name='integer_coordinates')
    # blindly assume the point is within our grid

    # https://github.com/tensorflow/tensorflow/pull/15857
    # "on CPU an error will be returned and on GPU 0 value will be filled to the expected positions of the output."
    # TODO: make this handle out of bounds correctly. I think correctly for us means return large number for SDF
    # and a gradient towards the origin

    # for coordinate in integer_coordinates:
    #     if c

    sdf_value = tf.gather_nd(sdf, integer_coordinates, name='sdf_gather')
    sdf_value = tf.reshape(sdf_value, [-1, 1], name='index_sdfs')

    def __sdf_gradient_func(dy):
        sdf_gradient = tf.gather_nd(full_sdf_gradient, integer_coordinates, name='sdf_gradients_gather')
        sdf_gradient = tf.reshape(sdf_gradient, [-1, P], name='index_sdf_gradient')
        return None, None, None, None, dy * sdf_gradient, None

    return sdf_value, __sdf_gradient_func
