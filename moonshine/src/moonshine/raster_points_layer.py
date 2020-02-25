import numpy as np
import tensorflow as tf

from link_bot_pycommon import link_bot_pycommon
from moonshine.action_smear_layer import smear_action
from moonshine.numpy_utils import add_batch


def raster(state, res, origin, h, w):
    """
    state: [batch, n]
    res: [batch] scalar float
    origin: [batch, 2] index (so int, or technically float is fine too)
    h: scalar int
    w: scalar int
    """
    b = state.shape[0]
    points = np.reshape(state, [b, -1, 2])
    n_points = points.shape[1]

    res = res[0]  # NOTE: assume constant resolution

    # points[:,1] is y, origin[0] is row index, so yes this is correct
    row_y_indices = (points[:, :, 1] / res + origin[:, 0:1]).astype(np.int64).flatten()
    col_x_indices = (points[:, :, 0] / res + origin[:, 1:2]).astype(np.int64).flatten()
    channel_indices = np.tile(np.arange(n_points), b)
    batch_indices = np.repeat(np.arange(b), n_points)

    # filter out invalid indices, which can happen during training
    rope_images = np.zeros([b, h, w, n_points], dtype=np.float32)
    valid_indices = np.where(np.all([row_y_indices >= 0,
                                     row_y_indices < h,
                                     col_x_indices >= 0,
                                     col_x_indices < w], axis=0))

    rope_images[batch_indices[valid_indices],
                row_y_indices[valid_indices],
                col_x_indices[valid_indices],
                channel_indices[valid_indices]] = 1.0
    return rope_images


def make_transition_image(local_env, planned_state, action, planned_next_state, res, origin, action_in_image: bool):
    """
    :param local_env: [h,w]
    :param planned_state: [n_state]
    :param action: [n_action]
    :param planned_next_state: [n_state]
    :param res: []
    :param origin: [2]
    :return: [n_points*2+n_action+1], aka  [n_state+n_action+1]
    """
    # TODO: make this operate on batched data, or make make_traj_image NOT operate on batched data
    h, w = local_env.shape
    local_env = np.expand_dims(local_env, axis=2)

    planned_rope_image = raster(*add_batch(planned_state, res, origin), h, w)[0]
    planned_next_rope_image = raster(*add_batch(planned_next_state, res, origin), h, w)[0]

    # action
    # add spatial dimensions and tile
    if action_in_image:
        image = np.concatenate((planned_rope_image, planned_next_rope_image, local_env), axis=2)
    else:
        action_image = smear_action(*add_batch(action), h, w)[0]
        image = np.concatenate((planned_rope_image, planned_next_rope_image, local_env, action_image), axis=2)
    return image


def raster_rope_images(planned_states, res, origins, h, w):
    """
    Raster all the state into one fixed-channel image representation using color gradient in the green channel
    :param planned_states: [batch, time, n_state]
    :param res: [batch]
    :param origins: [batch, 2]
    :param h: scalar
    :param w: scalar
    :return: [batch, time, h, w, 2]
    """
    b, n_time_steps, _ = planned_states.shape
    rope_images = np.zeros([b, h, w, 2], dtype=np.float32)
    for t in range(n_time_steps):
        planned_states_t = planned_states[:, t]
        rope_img_t = raster(planned_states_t, res, origins, h, w)
        rope_img_t = np.sum(rope_img_t, axis=3)
        gradient_t = float(t) / n_time_steps
        gradient_image_t = rope_img_t * gradient_t
        rope_images[:, :, :, 0] += rope_img_t
        rope_images[:, :, :, 1] += gradient_image_t
    rope_images = np.clip(rope_images, 0, 1.0)
    return rope_images


def make_traj_image(full_env, full_env_origin, res, states):
    """
    :param full_env: [batch, h, w]
    :param full_env_origin:  [batch, 2]
    :param res: [batch]
    :param states: [batch, time, n]
    :return: [batch, h, w, 3]
    """
    b, h, w = full_env.shape

    # add channel index
    full_env = np.expand_dims(full_env, axis=3)

    rope_imgs = raster_rope_images(states, res, full_env_origin, h, w)

    image = np.concatenate((full_env, rope_imgs), axis=3)
    return image


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
        self.n_points = link_bot_pycommon.n_state_to_n_points(self.n)

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
        row_indices = tf.reshape(row_y_indices, [-1])
        col_indices = tf.reshape(col_x_indices, [-1])
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
