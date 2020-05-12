from typing import Optional, Dict, List

import tensorflow as tf

from link_bot_data.link_bot_dataset_utils import NULL_PAD_VALUE, add_next_and_planned, add_planned, state_dict_is_null_tf, \
    total_state_dim
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.action_smear_layer import smear_action_differentiable
from moonshine.get_local_environment import get_local_env_and_origin_differentiable
from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts_tf


def make_state_and_env_image(environment: Dict,
                             state_dict: Dict,
                             scenario: ExperimentScenario,
                             local_env_h: int,
                             local_env_w: int,
                             k: float,
                             batch_size: Optional[int] = 1,
                             ):
    """
    :param environment:
    :param state_dict: each element should be [batch,n_state]
    :param scenario:
    :param local_env_h:
    :param local_env_w:
    :param k: constant controlling fuzzyness of how the rope is drawn in the image, should be like 1000
    :param batch_size:

    :return [batch,n_points*2+1], aka  [batch,n_state+1]
    """
    local_env_center_point = scenario.local_environment_center_differentiable(state_dict)
    local_env, local_env_origin = get_local_env_and_origin_differentiable(center_point=local_env_center_point,
                                                                          full_env=environment['full_env/env'],
                                                                          full_env_origin=environment['full_env/origin'],
                                                                          res=environment['full_env/res'],
                                                                          local_h_rows=local_env_h,
                                                                          local_w_cols=local_env_w)

    concat_args = []
    for planned_state in state_dict.values():
        planned_rope_image = raster_differentiable(state=planned_state,
                                                   res=environment['full_env/res'],
                                                   origin=local_env_origin,
                                                   h=local_env_h,
                                                   w=local_env_w,
                                                   k=k,
                                                   batch_size=batch_size)
        concat_args.append(planned_rope_image)

    concat_args.append(tf.expand_dims(local_env, axis=3))
    image = tf.concat(concat_args, axis=3)
    return image


def make_transition_images(environment: Dict,
                           state_dict: Dict,
                           action,
                           next_state_dict: Dict,
                           scenario: ExperimentScenario,
                           local_env_h: int,
                           local_env_w: int,
                           k: float,
                           batch_size: Optional[int] = 1,
                           action_in_image: Optional[bool] = False,
                           ):
    """
    :param environment:
    :param state_dict: each element should be [batch,n_state]
    :param action: [batch,n_action]
    :param next_state_dict: each element should be [batch,n_state]
    :param scenario:
    :param local_env_h:
    :param local_env_w:
    :param k: constant controlling fuzzyness of how the rope is drawn in the image, should be like 1000
    :param action_in_image: include new channels for actions
    :param batch_size:

    :return [batch,n_points*2+n_action+1], aka  [batch,n_state+n_action+1]
    """
    local_env_center_point = scenario.local_environment_center_differentiable(state_dict)
    local_env, local_env_origin = get_local_env_and_origin_differentiable(center_point=local_env_center_point,
                                                                          full_env=environment['full_env/env'],
                                                                          full_env_origin=environment['full_env/origin'],
                                                                          res=environment['full_env/res'],
                                                                          local_h_rows=local_env_h,
                                                                          local_w_cols=local_env_w)

    concat_args = []
    for planned_state in state_dict.values():
        planned_rope_image = raster_differentiable(state=planned_state,
                                                   res=environment['full_env/res'],
                                                   origin=local_env_origin,
                                                   h=local_env_h,
                                                   w=local_env_w,
                                                   k=k,
                                                   batch_size=batch_size)
        concat_args.append(planned_rope_image)
    for planned_next_state in next_state_dict.values():
        planned_next_rope_image = raster_differentiable(state=planned_next_state,
                                                        origin=local_env_origin,
                                                        res=environment['full_env/res'],
                                                        h=local_env_h,
                                                        w=local_env_w,
                                                        k=k,
                                                        batch_size=batch_size)
        concat_args.append(planned_next_rope_image)

    if action_in_image:
        action_image = smear_action_differentiable(action, local_env_h, local_env_w)
        concat_args.append(action_image)

    concat_args.append(tf.expand_dims(local_env, axis=3))
    image = tf.concat(concat_args, axis=3)
    return image


def partial_add_transition_image(states_keys,
                                 scenario,
                                 local_env_h: int,
                                 local_env_w: int,
                                 batch_size: int,
                                 rope_image_k: float):
    def _add_transition_image(example):
        action = example['action']
        environment = {
            'full_env/env': example['full_env/env'],
            'full_env/origin': example['full_env/origin'],
            'full_env/res': example['full_env/res'],
        }

        state_dict = {}
        next_state_dict = {}
        for k in states_keys:
            planned_state = example[add_planned(k)]
            planned_state_next = example[add_next_and_planned(k)]
            state_dict[add_planned(k)] = planned_state
            next_state_dict[add_next_and_planned(k)] = planned_state_next

        transition_images = make_transition_images(environment=environment,
                                                   state_dict=state_dict,
                                                   action=action,
                                                   next_state_dict=next_state_dict,
                                                   scenario=scenario,
                                                   local_env_h=local_env_h,
                                                   local_env_w=local_env_w,
                                                   k=rope_image_k,
                                                   batch_size=batch_size,
                                                   action_in_image=False)

        example['transition_image'] = transition_images
        return example

    return _add_transition_image


def make_traj_images(scenario: ExperimentScenario,
                     environment,
                     states_list: List[Dict],
                     local_env_h: int,
                     local_env_w: int,
                     rope_image_k: float,
                     batch_size: int):
    """
    :param scenario:
    :param environment:
    :param states_dict: each element is [batch, time, n]
    :param local_env_h:
    :param local_env_w:
    :param rope_image_k:
    :param batch_size:
    :return: [batch, time, h, w, 1 + n_points]
    """
    images = []
    state_dim = total_state_dim(states_list[0]) + 1
    for states_dict_t in states_list:
        if state_dict_is_null_tf(states_dict_t):
            image = tf.zeros([batch_size, local_env_h, local_env_w, state_dim])
        else:
            image = make_state_and_env_image(environment=environment,
                                             state_dict=states_dict_t,
                                             scenario=scenario,
                                             local_env_h=local_env_h,
                                             local_env_w=local_env_w,
                                             k=rope_image_k,
                                             batch_size=batch_size)
        images.append(image)
    all_images = tf.stack(images, axis=1)
    return all_images


def make_traj_images_from_states_list(scenario: ExperimentScenario,
                                      environment: Dict,
                                      states: List[Dict],
                                      local_env_w: int,
                                      local_env_h: int,
                                      rope_image_k: float,
                                      ):
    """
    :param environment:
    :param states: each element is [batch, time, n]
    :return: [batch, h, w, 3]
    :param rope_image_k: large constant controlling fuzzyness of rope drawing, like 1000
    """
    return make_traj_images(scenario=scenario,
                            environment=environment,
                            states_list=states,
                            local_env_h=local_env_h,
                            local_env_w=local_env_w,
                            rope_image_k=rope_image_k,
                            batch_size=1)


def partial_add_traj_image(states_keys: List[str],
                           scenario: ExperimentScenario,
                           batch_size: int,
                           local_env_h: int,
                           local_env_w: int,
                           rope_image_k: float):
    def _add_traj_image(example):
        return add_traj_image_to_example(scenario=scenario,
                                         example=example,
                                         states_keys=states_keys,
                                         local_env_h=local_env_h,
                                         local_env_w=local_env_w,
                                         rope_image_k=rope_image_k,
                                         batch_size=batch_size)

    return _add_traj_image


def add_traj_image_to_example(scenario: ExperimentScenario,
                              example,
                              states_keys: List[str],
                              local_env_w: int,
                              local_env_h: int,
                              rope_image_k: float,
                              batch_size: int):
    environment = {
        'full_env/env': example['full_env/env'],
        'full_env/origin': example['full_env/origin'],
        'full_env/res': example['full_env/res'],
    }
    environment.update(scenario.get_environment_from_example(example))

    planned_states_dict = {}
    for state_key in states_keys:
        states_all = example[add_planned(state_key)]
        planned_states_dict[state_key] = states_all

    planned_states_list = dict_of_sequences_to_sequence_of_dicts_tf(planned_states_dict, time_axis=1)
    image = make_traj_images(scenario=scenario,
                             environment=environment,
                             states_list=planned_states_list,
                             local_env_w=local_env_w,
                             local_env_h=local_env_h,
                             rope_image_k=rope_image_k,
                             batch_size=batch_size)

    example['trajectory_image'] = image
    return example


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


def setup_image_inputs(args, scenario, classifier_dataset, model_hparams):
    postprocess = None
    image_key = model_hparams['image_key']
    if image_key == 'transition_image':
        postprocess = partial_add_transition_image(states_keys=model_hparams['states_keys'],
                                                   scenario=scenario,
                                                   local_env_h=model_hparams['local_env_h_rows'],
                                                   local_env_w=model_hparams['local_env_w_cols'],
                                                   batch_size=args.batch_size,
                                                   rope_image_k=model_hparams['rope_image_k'])
        model_hparams['input_h_rows'] = model_hparams['local_env_h_rows']
        model_hparams['input_w_cols'] = model_hparams['local_env_w_cols']
    elif image_key == 'trajectory_image':
        postprocess = partial_add_traj_image(states_keys=model_hparams['states_keys'],
                                             scenario=scenario,
                                             local_env_h=model_hparams['local_env_h_rows'],
                                             local_env_w=model_hparams['local_env_w_cols'],
                                             batch_size=args.batch_size,
                                             rope_image_k=model_hparams['rope_image_k'])
        model_hparams['input_h_rows'] = classifier_dataset.full_env_params.h_rows
        model_hparams['input_w_cols'] = classifier_dataset.full_env_params.w_cols
    return postprocess, model_hparams


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
