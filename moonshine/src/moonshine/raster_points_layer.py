from typing import Optional, Dict

import numpy as np
import tensorflow as tf

from link_bot_pycommon import link_bot_pycommon
from moonshine.action_smear_layer import smear_action
from moonshine.numpy_utils import add_batch


def differentiable_get_local_env():
    pass


def differentiable_raster(state, res, origins, h, w):
    # TODO: gradients?
    """
    state: [batch, n]
    res: [batch] scalar float
    origins: [batch, 2] index (so int, or technically float is fine too)
    h: scalar int
    w: scalar int
    """
    b = state.shape[0]
    points = tf.reshape(state, [b, -1, 2])
    n_points = points.shape[1]

    res = res[0]

    beta = 50.0

    ## Below is a un-vectorized implementation, which is much easier to read and understand
    # rope_images = np.zeros([b, h, w, n_points], dtype=np.float32)
    # for batch_index in range(b):
    #     for point_idx in range(n_points):
    #         for row, col in np.ndindex(h, w):
    #             point_in_meters = points[batch_index, point_idx]
    #             pixel_center_in_meters = idx_to_point(row, col, res, origins[batch_index])
    #             squared_distance = np.sum(np.square(point_in_meters - pixel_center_in_meters))
    #             pixel_value = np.exp(-beta*squared_distance)
    #             rope_images[batch_index, row, col, point_idx] += pixel_value
    # rope_images = rope_images

    ## vectorized implementation

    # add h & w dimensions
    tiled_points = tf.expand_dims(tf.expand_dims(points, axis=1), axis=1)
    tiled_points = tf.tile(tiled_points, [1, h, w, 1, 1])
    pixel_row_indices = tf.range(0, h, dtype=tf.float32)
    pixel_col_indices = tf.range(0, w, dtype=tf.float32)
    # pixel_indices is b, n_points, 2
    pixel_indices = tf.stack(tf.meshgrid(pixel_row_indices, pixel_col_indices), axis=2)
    # add batch dim
    pixel_indices = tf.expand_dims(pixel_indices, axis=0)
    pixel_indices = tf.tile(pixel_indices, [b, 1, 1, 1])

    # shape [b, h, w, 2]
    pixel_centers = (pixel_indices - origins) * res

    # add n_points dim
    pixel_centers = tf.expand_dims(pixel_centers, axis=3)
    pixel_centers = tf.tile(pixel_centers, [1, 1, 1, n_points, 1])

    squared_distances = tf.reduce_sum(tf.square(pixel_centers - tiled_points), axis=4)
    pixel_values = tf.exp(-beta * squared_distances)
    rope_images = tf.reshape(pixel_values, [b, h, w, n_points])
    return rope_images


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

    res = res[0]

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


def make_transition_images(local_env: np.ndarray,
                           planned_states: Dict[str, np.ndarray],
                           action: np.ndarray,
                           planned_next_states: Dict[str, np.ndarray],
                           res: np.ndarray,
                           origin: np.ndarray,
                           action_in_image: Optional[bool] = False):
    """
    :param local_env: [batch,h,w]
    :param planned_states: each element should be [batch,n_state]
    :param action: [batch,n_action]
    :param planned_next_states: each element should be [batch,n_state]
    :param res: [batch]
    :param origin: [batch,2]
    :param action_in_image: include new channels for actions
    :return: [batch,n_points*2+n_action+1], aka  [batch,n_state+n_action+1]
    """
    b, h, w = local_env.shape
    local_env = np.expand_dims(local_env, axis=3)

    concat_args = [local_env]
    for planned_state in planned_states.values():
        planned_rope_image = raster(planned_state, res, origin, h, w)
        concat_args.append(planned_rope_image)
    for planned_next_state in planned_next_states.values():
        planned_next_rope_image = raster(planned_next_state, res, origin, h, w)
        concat_args.append(planned_next_rope_image)

    if action_in_image:
        # FIXME: use tf to make sure its differentiable
        action_image = smear_action(action, h, w)
        concat_args.append(action_image)
    image = np.concatenate(concat_args, axis=3)
    return image


def raster_rope_images(planned_states: Dict[str, np.ndarray],
                       res: np.ndarray,
                       origins: np.ndarray,
                       h: float,
                       w: float):
    """
    Raster all the state into one fixed-channel image representation using color gradient in the green channel
    :param planned_states: each element is [batch, time, n_state]
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


def make_traj_images(full_env: np.ndarray,
                     full_env_origin: np.ndarray,
                     res: np.ndarray,
                     states: Dict[str, np.ndarray]):
    """
    :param full_env: [batch, h, w]
    :param full_env_origin:  [batch, 2]
    :param res: [batch]
    :param states: each element is [batch, time, n]
    :return: [batch, h, w, 3]
    """
    b, h, w = full_env.shape

    # add channel index
    full_env = np.expand_dims(full_env, axis=3)

    rope_imgs = raster_rope_images(states, res, full_env_origin, h, w)

    image = np.concatenate((full_env, rope_imgs), axis=3)
    return image
