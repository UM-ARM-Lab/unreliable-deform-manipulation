import numpy as np
import tensorflow as tf


class RasterPoints(tf.keras.layers.Layer):

    def __init__(self, local_env_shape, **kwargs):
        super(RasterPoints, self).__init__(**kwargs)
        self.local_env_shape = local_env_shape
        self.n = None
        self.n_points = None
        self.sequence_length = None

    def build(self, input_shapes):
        super(RasterPoints, self).build(input_shapes)
        self.sequence_length = int(input_shapes[0][1])
        self.n = int(input_shapes[0][2])
        self.n_points = int(self.n // 2)

    def call(self, inputs, **kwargs):
        """
        :param x: [batch_size, sequence_length, n_points * 2], [batch_size, sequence_length, 2], [batch_size, sequence_length, 2]
        :return: local_env_shape
        """
        x, resolution, origin = inputs
        batch_size = tf.cast(tf.shape(x)[0], tf.int64)
        points = tf.reshape(x, [batch_size, self.sequence_length, self.n_points, 2], name='points_reshape')

        np_rope_images = np.zeros(
            [batch_size, self.sequence_length, self.local_env_shape[0], self.local_env_shape[1], self.n_points])

        row_y_indices = tf.reshape(tf.cast(points[:, :, :, 1] / resolution[:, :, 0:1] + origin[:, :, 0:1], tf.int64), [-1])
        col_x_indices = tf.reshape(tf.cast(points[:, :, :, 0] / resolution[:, :, 1:2] + origin[:, :, 1:2], tf.int64), [-1])
        batch_indices = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size), [-1, 1]), [1, self.n_points * self.sequence_length]),
                                   [-1])
        time_indices = tf.tile(
            tf.reshape(tf.tile(tf.reshape(tf.range(self.sequence_length, dtype=tf.int64), [-1, 1]), [1, self.n_points]), [-1]),
            [batch_size])
        row_indices = tf.squeeze(row_y_indices)
        col_indices = tf.squeeze(col_x_indices)
        point_channel_indices = tf.tile(tf.range(self.n_points, dtype=tf.int64), [batch_size * self.sequence_length])
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

        indices_tuple = tuple([i.numpy() for i in valid_indices])
        np_rope_images[indices_tuple] = 1
        rope_images = tf.convert_to_tensor(np_rope_images, dtype=tf.float32)

        return rope_images

    def get_config(self):
        config = {}
        config.update(super(RasterPoints, self).get_config())
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.local_env_shape[0], self.local_env_shape[1], self.n_points
