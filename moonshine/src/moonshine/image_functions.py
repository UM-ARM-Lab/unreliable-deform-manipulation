from typing import Dict, List

import tensorflow as tf

from link_bot_data.link_bot_dataset_utils import NULL_PAD_VALUE
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.action_smear_layer import smear_action_differentiable
from moonshine.get_local_environment import get_local_env_and_origin_differentiable as get_local_env
from moonshine.moonshine_utils import flatten_batch_and_time, sequence_of_dicts_to_dict_of_sequences


def make_traj_images(local_env_center_point_batch_time,
                     environment,
                     states_dict: Dict,
                     actions,
                     local_env_h: int,
                     local_env_w: int,
                     rope_image_k: float,
                     batch_size: int,
                     action_in_image: bool = True):
    """
    :return: [batch, time, h, w, 1 + n_points]
    """
    # First flatten batch & time
    states_dict_batch_time = flatten_batch_and_time(states_dict)
    local_env_center_point_batch_time = scenario.local_environment_center_differentiable(states_dict_batch_time)

    zero_pad_actions = tf.pad(actions, [[0, 0], [0, 1], [0, 0]])
    actions_batch_time = tf.reshape(zero_pad_actions, [-1] + actions.shape.as_list()[2:])
    time = int(zero_pad_actions.shape[1])
    batch_and_time = int(actions_batch_time.shape[0])
    env_batch_time = tf.tile(environment['full_env/env'], [time, 1, 1])
    env_origin_batch_time = tf.tile(environment['full_env/origin'], [time, 1])
    env_res_batch_time = tf.tile(environment['full_env/res'], [time])

    # this will produce images even for "null" data,
    # but are masked out in the RNN, and not actually used in the computation
    local_env_batch_time, local_env_origin_batch_time = get_local_env(center_point=local_env_center_point_batch_time,
                                                                      full_env=env_batch_time,
                                                                      full_env_origin=env_origin_batch_time,
                                                                      res=env_res_batch_time,
                                                                      local_h_rows=local_env_h,
                                                                      local_w_cols=local_env_w)

    concat_args = []
    for planned_state in states_dict_batch_time.values():
        planned_rope_image = raster_differentiable(state=planned_state,
                                                   res=env_res_batch_time,
                                                   origin=local_env_origin_batch_time,
                                                   h=local_env_h,
                                                   w=local_env_w,
                                                   k=rope_image_k,
                                                   batch_size=batch_and_time)
        concat_args.append(planned_rope_image)

    if action_in_image:
        action_image = smear_action_differentiable(actions_batch_time, local_env_h, local_env_w)
        concat_args.append(action_image)

    concat_args.append(tf.expand_dims(local_env_batch_time, axis=3))
    images_batch_time = tf.concat(concat_args, axis=3)
    images = tf.reshape(images_batch_time, [batch_size, time, local_env_h, local_env_w, -1])
    return images


def make_traj_images_from_states_list(environment: Dict,
                                      states: List[Dict],
                                      actions,
                                      scenario: ExperimentScenario,
                                      local_env_w: int,
                                      local_env_h: int, rope_image_k: float):
    """
    :param environment:
    :param states: each element is [batch, time, n]
    :return: [batch, h, w, 3]
    :param rope_image_k: large constant controlling fuzzyness of rope drawing, like 1000
    """
    states_dict = sequence_of_dicts_to_dict_of_sequences(states)
    return make_traj_images(scenario=scenario,
                            environment=environment,
                            states_dict=states_dict,
                            actions=actions,
                            local_env_h=local_env_h,
                            local_env_w=local_env_w,
                            rope_image_k=rope_image_k,
                            batch_size=1)


@tf.function
def raster_differentiable(state, res, origin, h, w, k, batch_size: int):
    """
    Even though this data is batched, we use singular and reserve plural for sequences in time
    Args:
        state: [batch, n]
        res: [batch] scalar float
        origin: [batch, 2] index (so int, or technically float is fine too)
        h: scalar int
        w: scalar int
        k: scalar float, should be very large, like 1000

    Returns:
     [batch, h, w, n_points]
    """

    res = res[0]
    n_points = int(int(state.shape[1]) / 2)
    points = tf.reshape(state, [batch_size, n_points, 2])

    ## Below is a un-vectorized implementation, which is much easier to read and understand
    # rope_images = np.zeros([batch_size, h, w, n_points], dtype=np.float32)
    # for batch_index in range(b):
    #     for point_idx in range(n_points):
    #         for row, col in np.ndindex(h, w):
    #             point_in_meters = points[batch_index, point_idx]
    #             pixel_center_in_meters = idx_to_point(row, col, res, origins[batch_index])
    #             squared_distance = np.sum(np.square(point_in_meters - pixel_center_in_meters))
    #             pixel_value = np.exp(-k*squared_distance)
    #             rope_images[batch_index, row, col, point_idx] += pixel_value
    # rope_images = rope_images

    ## vectorized implementation

    # add h & w dimensions
    tiled_points = tf.expand_dims(tf.expand_dims(points, axis=1), axis=1)
    tiled_points = tf.tile(tiled_points, [1, h, w, 1, 1])
    tiled_points_y_x = tf.reverse(tiled_points, axis=[4])
    pixel_row_indices = tf.range(0, h, dtype=tf.float32)
    pixel_col_indices = tf.range(0, w, dtype=tf.float32)
    # pixel_indices is batch_size, n_points, 2
    pixel_indices_row_col = tf.stack(tf.meshgrid(pixel_row_indices, pixel_col_indices), axis=2)
    # add batch dim
    pixel_indices_row_col = tf.expand_dims(pixel_indices_row_col, axis=0)
    pixel_indices_row_col = tf.tile(pixel_indices_row_col, [batch_size, 1, 1, 1])

    # shape [batch_size, h, w, 2]
    origin_expanded = tf.expand_dims(tf.expand_dims(origin, axis=1), axis=1)
    pixel_centers_y_x = (pixel_indices_row_col - origin_expanded) * res

    # add n_points dim
    pixel_centers_y_x = tf.expand_dims(pixel_centers_y_x, axis=3)
    pixel_centers_y_x = tf.tile(pixel_centers_y_x, [1, 1, 1, n_points, 1])

    squared_distances = tf.reduce_sum(tf.square(pixel_centers_y_x - tiled_points_y_x), axis=4)
    pixel_values = tf.exp(-k * squared_distances)
    rope_images = tf.transpose(tf.reshape(pixel_values, [batch_size, h, w, n_points]), [0, 2, 1, 3])
    ########################################################################################
    # TODO: figure out whether to do clipping or normalization, right now we don't do either
    ########################################################################################
    return rope_images


# Numpy is only used be the one function below
import numpy as np


def old_raster_wrapped(state, res, origin, h, w, k):
    rope_images = tf.numpy_function(old_raster, [state, res, origin, h, w], tf.float32)
    rope_images.set_shape([state.shape[0], h, w, 1])
    return rope_images


def old_raster(state, res, origin, h, w):
    """
    state: [batch, n]
    res: [batch] scalar float
    origin: [batch, 2] index (so int, or technically float is fine too)
    h: scalar int
    w: scalar int
    return: [batch, h, w, n_points]
    """
    b = int(state.shape[0])
    points = np.reshape(state, [b, -1, 2])
    n_points = int(points.shape[1])

    # FIXME: PERFORMANCE HACK
    if state[0, 0] == NULL_PAD_VALUE:
        empty_image = np.zeros([b, h, w, n_points], dtype=np.float32)
        return empty_image

    res = res[0]  # NOTE: assume constant resolution

    # points[:,1] is y, origin[0] is row index, so yes this is correct
    row_y_indices = (points[:, :, 1] / res + origin[:, 0:1]).astype(np.int64).flatten()
    col_x_indices = (points[:, :, 0] / res + origin[:, 1:2]).astype(np.int64).flatten()
    channel_indices = np.tile(np.arange(n_points), b)
    batch_indices = np.repeat(np.arange(b), n_points)

    # filter out invalid indices, which can happen during training
    state_images = np.zeros([b, h, w, n_points], dtype=np.float32)
    valid_indices = np.where(np.all([row_y_indices >= 0,
                                     row_y_indices < h,
                                     col_x_indices >= 0,
                                     col_x_indices < w], axis=0))

    state_images[batch_indices[valid_indices],
                 row_y_indices[valid_indices],
                 col_x_indices[valid_indices],
                 channel_indices[valid_indices]] = 1.0
    return state_images
