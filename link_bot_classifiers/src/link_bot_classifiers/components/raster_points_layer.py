import tensorflow as tf
import tensorflow.contrib.eager as tfe


class RasterPoints(tf.keras.layers.Layer):

    def __init__(self, sdf_shape, **kwargs):
        super(RasterPoints, self).__init__(**kwargs)
        self.sdf_shape = sdf_shape
        self.n = None
        self.n_points = None

    def build(self, input_shapes):
        super(RasterPoints, self).build(input_shapes)
        self.n = int(input_shapes[0][1])
        self.n_points = int(self.n // 2)

    def call(self, inputs, **kwargs):
        """
        :param x: [sequence_length, n_points * 2], [sequence_length, 2], [sequence_length, 2]
        :return: sdf_shape
        """
        x, resolution, origin = inputs
        points = tf.reshape(x, [-1, self.n_points, 2], name='points_reshape')

        batch_size = tf.cast(tf.shape(x)[0], tf.int64)
        rope_images = tfe.Variable(
            initial_value=lambda: tf.zeros([batch_size, self.sdf_shape[0], self.sdf_shape[1], self.n_points]),
            name='rope_images',
            trainable=False)
        row_y_indeces = tf.cast(points[:, :, 1] / resolution[:, 0:1] + origin[:, 0:1], tf.int64)
        col_x_indeces = tf.cast(points[:, :, 0] / resolution[:, 1:2] + origin[:, 1:2], tf.int64)
        batch_indeces = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size), [-1, 1]), [1, self.n_points]), [-1])
        row_indeces = tf.squeeze(row_y_indeces)
        col_indeces = tf.squeeze(col_x_indeces)
        point_channel_indeces = tf.tile(tf.range(self.n_points, dtype=tf.int64), [batch_size])
        indices = tf.transpose(tf.stack([batch_indeces, row_indeces, col_indeces, point_channel_indeces]))
        on_pixels = tf.gather_nd(rope_images, indices)
        # rope_images = tf.assign(rope_images[batch_indeces, row_indeces, col_indeces, point_channel_indeces], 1)
        # tf.assign(on_pixels, 1)

        ones = tf.ones_like(indices[:, 0], dtype=tf.float32)
        rope_images = tf.sparse.SparseTensor(indices=indices,
                                             values=ones,
                                             dense_shape=[batch_size, self.sdf_shape[0], self.sdf_shape[1], self.n_points])
        rope_images = tf.sparse.to_dense(rope_images)
        return rope_images

    def get_config(self):
        config = {}
        config.update(super(RasterPoints, self).get_config())
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.sdf_shape[0], self.sdf_shape[1], self.n_points
