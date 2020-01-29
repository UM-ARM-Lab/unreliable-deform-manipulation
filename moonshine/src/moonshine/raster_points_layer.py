import numpy as np
import tensorflow as tf


class RasterPoints(tf.keras.layers.Layer):

    def __init__(self, local_env_shape, batch_size, **kwargs):
        super(RasterPoints, self).__init__(**kwargs)
        self.local_env_shape = local_env_shape
        self.n = None
        self.n_points = None
        self.sequence_length = None
        self.batch_size = np.int64(batch_size)

    def build(self, input_shapes):
        super(RasterPoints, self).build(input_shapes)
        self.sequence_length = int(input_shapes[0][1])
        self.n = int(input_shapes[0][2])
        self.n_points = int(self.n // 2)

    def call(self, inputs, **kwargs):
        """
        :param inputs:
            x: [batch_size, sequence_length, n_points * 2], float
            resolution: [batch_size, sequence_length, 2], float
            origin: [batch_size, sequence_length, 2], float
        :return: local_env_shape
        """
        x, resolution, origin = inputs
        points = tf.reshape(x, [self.batch_size, self.sequence_length, self.n_points, 2], name='points_reshape')


        # resolution is assumed to be x,y, origin is row,col (which is y,x)
        row_y_indices = tf.reshape(tf.cast(points[:, :, :, 1] / resolution[:, :, 1:2] + origin[:, :, 0:1], tf.int64), [-1])
        col_x_indices = tf.reshape(tf.cast(points[:, :, :, 0] / resolution[:, :, 0:1] + origin[:, :, 1:2], tf.int64), [-1])
        batch_indices = tf.reshape(
            tf.tile(tf.reshape(tf.range(self.batch_size), [-1, 1]), [1, self.n_points * self.sequence_length]),
            [-1])
        time_indices = tf.tile(
            tf.reshape(tf.tile(tf.reshape(tf.range(self.sequence_length, dtype=tf.int64), [-1, 1]), [1, self.n_points]), [-1]),
            [self.batch_size])
        row_indices = tf.squeeze(row_y_indices)
        col_indices = tf.squeeze(col_x_indices)
        point_channel_indices = tf.tile(tf.range(self.n_points, dtype=tf.int64), [self.batch_size * self.sequence_length])
        indices = tf.stack((batch_indices,
                            time_indices,
                            row_indices,
                            col_indices,
                            point_channel_indices), axis=1)

        # filter out any invalid indices
        in_bounds_row = tf.logical_and(tf.greater_equal(indices[:, 2], 0), tf.less(indices[:, 2], self.local_env_shape[0]))
        in_bounds_col = tf.logical_and(tf.greater_equal(indices[:, 3], 0), tf.less(indices[:, 3], self.local_env_shape[1]))
        in_bounds = tf.math.reduce_all(tf.stack((in_bounds_row, in_bounds_col), axis=1), axis=1)
        valid_indices = tf.boolean_mask(indices, in_bounds)
        valid_indices = tf.unstack(valid_indices, axis=1)

        output_shape = [self.batch_size, self.sequence_length, self.local_env_shape[0], self.local_env_shape[1], self.n_points]

        def _index(*valid_indices):
            np_rope_images = np.zeros(output_shape, dtype=np.float32)
            np_rope_images[tuple(valid_indices)] = 1
            return np_rope_images

        rope_images = tf.numpy_function(_index, inp=valid_indices, Tout=tf.float32)
        rope_images.set_shape(output_shape)

        return rope_images

    def get_config(self):
        config = {}
        config.update(super(RasterPoints, self).get_config())
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.local_env_shape[0], self.local_env_shape[1], self.n_points
