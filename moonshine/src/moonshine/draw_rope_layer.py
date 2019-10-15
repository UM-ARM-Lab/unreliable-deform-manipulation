import numpy as np
import tensorflow as tf
from bresenham import bresenham

from link_bot_pycommon.link_bot_sdf_utils import point_to_idx


class DrawRope(tf.keras.layers.Layer):

    def __init__(self, image_shape, **kwargs):
        super(DrawRope, self).__init__(**kwargs)
        self.image_shape = image_shape
        self.n = None
        self.n_points = None
        self.sequence_length = None

    def build(self, input_shapes):
        super(DrawRope, self).build(input_shapes)
        self.sequence_length = int(input_shapes[0][1])
        self.n = int(input_shapes[0][2])
        self.n_points = int(self.n // 2)

    def call(self, inputs, **kwargs):
        """
        :param x: [sequence_length, n_points * 2], [sequence_length, 2], [sequence_length, 2]
        :return: local_env_shape
        """
        x, resolution, origin = inputs
        batch_size = int(tf.cast(tf.shape(x)[0], tf.int64))
        points = tf.reshape(x, [batch_size, self.sequence_length, self.n_points, 2], name='points_reshape')

        rope_images = tf.py_func(self.draw_ropes, [points, resolution, origin], tf.uint8)
        rope_images.set_shape([batch_size, self.sequence_length, self.image_shape[0], self.image_shape[1], self.n_points])
        return rope_images

    def draw_ropes(self, batch_points, resolution, origin):
        batch_size = batch_points.shape[0]
        rope_images = np.zeros([batch_size, self.sequence_length, self.image_shape[0], self.image_shape[1], 3], dtype=np.uint8)
        for batch_idx, points in enumerate(batch_points):
            for t_idx, points_t in enumerate(points):
                r0, c0 = point_to_idx(points_t[0, 0], points_t[0, 1], resolution, origin)
                r1, c1 = point_to_idx(points_t[1, 0], points_t[1, 1], resolution, origin)
                r2, c2 = point_to_idx(points_t[2, 0], points_t[2, 1], resolution, origin)
                for c, r in bresenham(c0, r0, c1, r1):
                    rope_images[batch_idx, t_idx, r, c, :] = 128
                for c, r in bresenham(c1, r1, c2, r2):
                    rope_images[batch_idx, t_idx, r, c, :] = 128
                rope_images[batch_idx, t_idx, r0, c0] = [255, 0, 0]
                rope_images[batch_idx, t_idx, r1, c1] = [255, 0, 0]
                rope_images[batch_idx, t_idx, r2, c2] = [0, 255, 0]
        return rope_images

    def get_config(self):
        config = {}
        config.update(super(DrawRope, self).get_config())
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.image_shape[0], self.image_shape[1], self.n_points
