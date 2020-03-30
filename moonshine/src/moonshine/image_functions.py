import re
from typing import Optional, Dict, List

import tensorflow as tf
from link_bot_data.link_bot_dataset_utils import add_all, add_planned

from link_bot_planning.experiment_scenario import ExperimentScenario
from link_bot_pycommon import link_bot_pycommon
from moonshine.get_local_environment import get_local_env_and_origin_differentiable
from moonshine.action_smear_layer import smear_action_differentiable
from moonshine.numpy_utils import add_batch


# @tf.function
def make_transition_image(full_env,
                          full_env_origin,
                          res,
                          planned_states,
                          action,
                          planned_next_states,
                          scenario: ExperimentScenario,
                          local_env_h: int,
                          local_env_w: int,
                          k: float,
                          action_in_image: Optional[bool] = False):
    """
    Args:
        full_env:
        full_env_origin:
        res: [batch]
        planned_states: each element should be [batch,n_state]
        action: [batch,n_action]
        planned_next_states: each element should be [batch,n_state]
        scenario:
        local_env_h:
        local_env_w:
        k: constant controlling fuzzyness of how the rope is drawn in the image, should be like 1000
        action_in_image: include new channels for actions

    Return:
        [batch,n_points*2+n_action+1], aka  [batch,n_state+n_action+1]
    """
    # somehow fix this, it's showing the rope not at the center of the local environment?!
    local_env_center_point = scenario.local_environment_center_differentiable(planned_states)
    local_env, local_env_origin = get_local_env_and_origin_differentiable(center_point=local_env_center_point,
                                                                          full_env=full_env,
                                                                          full_env_origin=full_env_origin,
                                                                          res=res,
                                                                          local_h_rows=local_env_h,
                                                                          local_w_cols=local_env_w)

    concat_args = []
    for planned_state in planned_states.values():
        planned_rope_image = raster_differentiable(state=planned_state,
                                                   res=res,
                                                   origin=local_env_origin,
                                                   h=local_env_h,
                                                   w=local_env_w,
                                                   k=k)
        concat_args.append(planned_rope_image)
    for planned_next_state in planned_next_states.values():
        planned_next_rope_image = raster_differentiable(state=planned_next_state,
                                                        origin=local_env_origin,
                                                        res=res,
                                                        h=local_env_h,
                                                        w=local_env_w,
                                                        k=k)
        concat_args.append(planned_next_rope_image)

    if action_in_image:
        action_image = smear_action_differentiable(action, local_env_h, local_env_w)
        concat_args.append(action_image)

    concat_args.append(tf.expand_dims(local_env, axis=3))
    image = tf.concat(concat_args, axis=3)
    return image


# @tf.function
def raster_rope_images(planned_states: Dict,
                       res,
                       origin,
                       h: float,
                       w: float,
                       k: float):
    """
    Raster all the states into one image representation using binary in the first set and gradient in the second set
        planned_states: each element is [batch, time, n_state]
    Args:
        res: [batch]
        origin: [batch, 2]
        h: scalar
        w: scalar
        k: scalar float, should be very large, like 1000
    Return:
        [batch, h, w, 2 * n_points]
    """
    binary_rope_images = []
    time_colored_rope_images = []

    for planned_state_seq in planned_states.values():
        n_time_steps = int(planned_state_seq.shape[1])
        for t in range(n_time_steps):
            planned_state_t = planned_state_seq[:, t]
            # should have shape batch, h, w, n_points
            rope_img_t = raster_differentiable(state=planned_state_t, origin=origin, res=res, h=h, w=w, k=k)
            time_color = float(t) / n_time_steps
            time_color_image_t = rope_img_t * time_color
            binary_rope_images.append(rope_img_t)
            time_colored_rope_images.append(time_color_image_t)

    binary_rope_images = tf.reduce_sum(binary_rope_images, axis=0)
    time_colored_rope_images = tf.reduce_sum(time_colored_rope_images, axis=0)
    rope_images = tf.concat((binary_rope_images, time_colored_rope_images), axis=3)
    return rope_images


def make_traj_images(full_env,
                     full_env_origin,
                     res,
                     states: List[Dict],
                     rope_image_k: float):
    """
    :param full_env: [batch, h, w]
    :param full_env_origin:  [batch, 2]
    :param res: [batch]
    :param states: each element is [batch, time, n]
    :return: [batch, h, w, 3]
    :param rope_image_k: large constant controlling fuzzyness of rope drawing, like 1000
    """
    # Reformat the list of dicts of tensors into one dict of tensors
    T = len(states)
    states_dict = {}
    keys = states[0].keys()
    for key in keys:
        state_with_time = []
        for t in range(T):
            state_with_time.append(states[t][key])
        state_with_time = tf.stack(state_with_time, axis=1)
        states_dict[key] = state_with_time

    return make_traj_images_with_dict(full_env=full_env,
                                      full_env_origin=full_env_origin,
                                      res=res,
                                      states_dict=states_dict,
                                      rope_image_k=rope_image_k)


# @tf.function
def make_traj_images_with_dict(full_env,
                               full_env_origin,
                               res,
                               states_dict: Dict,
                               rope_image_k: float):
    """
    :param full_env: [batch, h, w]
    :param full_env_origin:  [batch, 2]
    :param res: [batch]
    :param states_dict: each element is [batch, time, n]
    :return: [batch, h, w, 3]
    """
    h = int(full_env.shape[1])
    w = int(full_env.shape[2])

    # add channel index
    full_env = tf.expand_dims(full_env, axis=3)

    rope_imgs = raster_rope_images(states_dict, res, full_env_origin, h, w, k=rope_image_k)

    image = tf.concat((full_env, rope_imgs), axis=3)
    return image


# @tf.function
def add_traj_image_wrapper(input_dict, states_keys: List[str], rope_image_k: float):
    full_env = input_dict['full_env/env']
    full_env_origin = input_dict['full_env/origin']
    res = input_dict['full_env/res']

    planned_states_dict = {}
    for state_key in states_keys:
        states_all = input_dict[add_all(add_planned(state_key))]
        planned_states_dict[state_key] = tf.expand_dims(states_all, axis=0)

    full_env, full_env_origin, res = add_batch(full_env, full_env_origin, res)

    image = make_traj_images_with_dict(full_env=full_env,
                                       full_env_origin=full_env_origin,
                                       res=res,
                                       states_dict=planned_states_dict,
                                       rope_image_k=rope_image_k)[0]

    input_dict['trajectory_image'] = image
    return input_dict


def add_traj_image(dataset, states_keys: List[str], rope_image_k: float):
    # @tf.function
    def _add_traj_image_wrapper(input_dict):
        return add_traj_image_wrapper(input_dict, states_keys, rope_image_k=rope_image_k)

    return dataset.map(_add_traj_image_wrapper)


def add_transition_image_to_example(input_dict,
                                    states_keys: List[str],
                                    scenario: ExperimentScenario,
                                    local_env_w: int,
                                    local_env_h: int,
                                    rope_image_k: float,
                                    action_in_image: Optional[bool] = False):
    action = input_dict['action']

    planned_states = {}
    planned_next_states = {}
    n_total_points = 0
    for state_key in states_keys:
        planned_state_feature_name = 'planned_state/{}'.format(state_key)
        planned_state_next_feature_name = 'planned_state/{}_next'.format(state_key)
        planned_state = input_dict[planned_state_feature_name]
        planned_next_state = input_dict[planned_state_next_feature_name]
        n_total_points += link_bot_pycommon.n_state_to_n_points(planned_state.shape[0])
        planned_states[state_key] = planned_state
        planned_next_states[state_key] = planned_next_state

    full_env = input_dict['full_env/env']
    full_env_res = tf.squeeze(input_dict['full_env/res'])
    full_env_origin = input_dict['full_env/origin']
    n_action = action.shape[0]

    batched_inputs = add_batch(full_env, full_env_origin, full_env_res, planned_states, action, planned_next_states)
    image = make_transition_image(*batched_inputs,
                                  scenario=scenario,
                                  local_env_h=local_env_h,
                                  local_env_w=local_env_w,
                                  k=rope_image_k,
                                  action_in_image=action_in_image)
    # remove batch dim
    image = image[0]
    n_channels = 1 + 2 * n_total_points
    if action_in_image:
        n_channels += n_action

    image.set_shape([local_env_h, local_env_w, n_channels])

    input_dict['transition_image'] = image
    return input_dict


def add_transition_image(dataset,
                         states_keys: List[str],
                         scenario: ExperimentScenario,
                         local_env_w: int,
                         local_env_h: int,
                         rope_image_k: float,
                         action_in_image: Optional[bool] = False):
    def _add_transition_image(input_dict):
        return add_transition_image_to_example(input_dict=input_dict,
                                               states_keys=states_keys,
                                               scenario=scenario,
                                               local_env_w=local_env_w,
                                               local_env_h=local_env_h,
                                               rope_image_k=rope_image_k,
                                               action_in_image=action_in_image)

    return dataset.map(_add_transition_image)


# @tf.function
def raster_differentiable(state, res, origin, h, w, k):
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

    b = int(state.shape[0])
    res = res[0]
    points = tf.reshape(state, [b, -1, 2])
    n_points = int(points.shape[1])

    ## Below is a un-vectorized implementation, which is much easier to read and understand
    # rope_images = np.zeros([b, h, w, n_points], dtype=np.float32)
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
    # pixel_indices is b, n_points, 2
    pixel_indices_row_col = tf.stack(tf.meshgrid(pixel_row_indices, pixel_col_indices), axis=2)
    # add batch dim
    pixel_indices_row_col = tf.expand_dims(pixel_indices_row_col, axis=0)
    pixel_indices_row_col = tf.tile(pixel_indices_row_col, [b, 1, 1, 1])

    # shape [b, h, w, 2]
    origin_expanded = tf.expand_dims(tf.expand_dims(origin, axis=1), axis=1)
    pixel_centers_y_x = (pixel_indices_row_col - origin_expanded) * res

    # add n_points dim
    pixel_centers_y_x = tf.expand_dims(pixel_centers_y_x, axis=3)
    pixel_centers_y_x = tf.tile(pixel_centers_y_x, [1, 1, 1, n_points, 1])

    squared_distances = tf.reduce_sum(tf.square(pixel_centers_y_x - tiled_points_y_x), axis=4)
    pixel_values = tf.exp(-k * squared_distances)
    rope_images = tf.transpose(tf.reshape(pixel_values, [b, h, w, n_points]), [0, 2, 1, 3])
    ########################################################################################
    # TODO: figure out whether to do clipping or normalization, right now we don't do either
    ########################################################################################
    return rope_images


# Numpy is only used be the one function below
import numpy as np


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
