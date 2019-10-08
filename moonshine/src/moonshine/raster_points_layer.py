import tensorflow as tf
import tensorflow.contrib.eager as tfe


class RasterPoints(tf.keras.layers.Layer):

    def __init__(self, sdf_shape, **kwargs):
        super(RasterPoints, self).__init__(**kwargs)
        self.sdf_shape = sdf_shape
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
        :return: sdf_shape
        """
        x, resolution, origin = inputs
        batch_size = tf.cast(tf.shape(x)[0], tf.int64)
        points = tf.reshape(x, [batch_size, self.sequence_length, self.n_points, 2], name='points_reshape')

        rope_images = tfe.Variable(
            initial_value=lambda: tf.zeros(
                [batch_size, self.sequence_length, self.sdf_shape[0], self.sdf_shape[1], self.n_points]),
            name='rope_images',
            trainable=False)
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
        indices = zip(batch_indices, time_indices, row_indices, col_indices, point_channel_indices)
        for batch_idx, time_idx, row_idx, col_idx, channel_idx in indices:
            rope_images = rope_images[batch_idx, time_idx, row_idx, col_idx, channel_idx].assign(1.0)

        return rope_images

    def get_config(self):
        config = {}
        config.update(super(RasterPoints, self).get_config())
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.sdf_shape[0], self.sdf_shape[1], self.n_points
