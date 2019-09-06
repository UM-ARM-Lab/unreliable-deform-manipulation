import numpy as np
import tensorflow as tf


class RasterPoints(tf.keras.layers.Layer):

    def __init__(self, sdf_shape, **kwargs):
        super(RasterPoints, self).__init__(**kwargs)
        self.sdf_shape = sdf_shape
        self.n = None
        self.n_points = None

    def build(self, input_shapes):
        super(RasterPoints, self).build(input_shapes)
        self.n = int(input_shapes[0][1])
        self.n_points = self.n // 2

    def raster_points(self, rope_configurations, resolution, origin):
        rope_configurations = np.atleast_2d(rope_configurations)
        batch_size, _, _ = rope_configurations.shape
        rope_images = np.zeros([batch_size, self.sdf_shape[0], self.sdf_shape[1], self.n_points], dtype=np.float32)
        batch_resolution = np.expand_dims(resolution, axis=1)
        batch_origin = np.expand_dims(origin, axis=1)
        indeces = (rope_configurations / batch_resolution + batch_origin).astype(np.int64)
        batch_indeces = np.arange(batch_size).repeat(self.n_points)
        row_indeces = indeces[:, :, 1].flatten()
        col_indeces = indeces[:, :, 0].flatten()
        point_channel_indeces = np.tile(np.arange(self.n_points), batch_size)
        if np.any(rope_configurations > 0.5) or np.any(rope_configurations < -0.5):
            raise ValueError(np.array2string(rope_configurations))
        rope_images[batch_indeces, row_indeces, col_indeces, point_channel_indeces] = 1
        return rope_images

    def call(self, inputs, **kwargs):
        """
        :param x: [n_points * 2]
        :return: sdf_shape
        """
        x, resolution, origin = inputs
        points = tf.reshape(x, [-1, self.n_points, 2])
        rope_image = tf.py_func(self.raster_points, [points, resolution, origin], tf.float32, name='raster_points')
        input_shapes = [input.shape for input in inputs]
        rope_image.set_shape(self.compute_output_shape(input_shapes))
        return rope_image

    def get_config(self):
        config = {}
        config.update(super(RasterPoints, self).get_config())
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.sdf_shape[0], self.sdf_shape[1], self.n_points
