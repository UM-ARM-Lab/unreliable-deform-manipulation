import tensorflow as tf


def ravel_2d(indeces_batch, cols):
    return indeces_batch[:, 0] * cols + indeces_batch[:, 1]


@tf.custom_gradient
def sdf_func(inputs, sdf_shape):
    sdf_size = sdf_shape[0] * sdf_shape[1]
    gradient_size = sdf_size * 2

    start_sdf = 0
    end_sdf = sdf_size
    end_gradient = end_sdf + gradient_size
    end_resolution = end_gradient + 2
    end_origin = end_resolution + 2
    end_input_point = end_origin + 2

    sdf_flat = inputs[:, start_sdf:end_sdf]
    gradient_flat = inputs[:, end_sdf:end_gradient]
    gradient_flat = tf.reshape(gradient_flat, [-1, sdf_size, 2])
    resolution = inputs[:, end_gradient:end_resolution]
    origin = inputs[:, end_resolution:end_origin]
    input_points = inputs[:, end_origin:end_input_point]

    float_input_points = tf.math.divide(input_points, resolution, name='float_input_points')
    float_offset_input_points = tf.add(float_input_points, origin, name='float_offset_input_points')
    integer_input_points = tf.cast(float_offset_input_points, dtype=tf.int32)
    integer_input_points_flat = ravel_2d(integer_input_points, sdf_shape[1])
    integer_input_points_flat = tf.expand_dims(integer_input_points_flat, axis=1, name='integer_input_points_flat')

    oob_left = integer_input_points[:, 0] < 0
    oob_right = integer_input_points[:, 0] >= sdf_shape[1]
    oob_up = integer_input_points[:, 1] < 0
    oob_down = integer_input_points[:, 1] >= sdf_shape[0]
    out_of_bounds = tf.math.reduce_any(tf.stack((oob_up, oob_down, oob_left, oob_right), axis=1), axis=1)

    sdf_value = tf.batch_gather(sdf_flat, integer_input_points_flat, name='sdf_gather')

    # TODO: how to handle out of bounds??? current set to slightly inside an obstacle
    oob_value = tf.ones_like(sdf_value) * -0.1
    sdf_value = tf.where(out_of_bounds, oob_value, sdf_value, name='sdf_value')

    def __sdf_gradient_func(dy):
        sdf_gradient = tf.batch_gather(gradient_flat, integer_input_points_flat, name='sdf_gradients_gather')
        sdf_gradient = tf.reshape(sdf_gradient, [-1, 2])
        full_zeros = tf.zeros_like(inputs)
        zero_padding = full_zeros[:, :end_origin]
        full_sdf_gradient = tf.concat([zero_padding, sdf_gradient], axis=1)

        # TODO: how to handle out of bounds??? current just zero everything
        # oob_gradient = tf.ones_like(sdf_gradient) * 0.0
        # sdf_gradient = tf.where(out_of_bounds, oob_gradient, sdf_gradient, name='sdf_gradient')

        return dy * full_sdf_gradient, None

    return sdf_value, __sdf_gradient_func


class SDFLookup(tf.keras.layers.Layer):

    def __init__(self, sdf_shape, **kwargs):
        self.sdf_shape = sdf_shape
        super(SDFLookup, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SDFLookup, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        return sdf_func(x, self.sdf_shape)

    def get_config(self):
        config = {'sdf_shape': self.sdf_shape}
        config.update(super(SDFLookup, self).get_config())
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1
