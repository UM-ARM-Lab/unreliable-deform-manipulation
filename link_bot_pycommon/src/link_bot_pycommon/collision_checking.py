from typing import Dict

import numpy as np
import tensorflow as tf

from link_bot_pycommon.grid_utils import point_to_idx, OccupancyData, batch_point_to_idx_tf, batch_point_to_idx_tf_3d
from moonshine.gpu_config import limit_gpu_mem


def batch_out_of_bounds_tf(environment: Dict,
                           xs: float,
                           ys: float):
    origin = environment['origin']
    res = environment['res']
    env = environment['env']
    h, w, _ = env.shape
    gripper_rows, gripper_cols = batch_point_to_idx_tf(xs, ys, res, origin)
    out_of_bounds = tf.reduce_any(gripper_rows >= h)
    out_of_bounds = out_of_bounds or tf.reduce_any(gripper_rows < 0)
    out_of_bounds = out_of_bounds or tf.reduce_any(gripper_cols >= w)
    out_of_bounds = out_of_bounds or tf.reduce_any(0 > gripper_cols)
    return out_of_bounds


def batch_in_collision_tf_3d(environment: Dict,
                             xs,
                             ys,
                             zs,
                             inflate_radius_m: float,
                             occupied_threshold: float = 0.5):
    origin = environment['origin']
    res = environment['res']
    env = environment['env']
    rows, cols, channels = batch_point_to_idx_tf_3d(xs, ys, zs, res, origin)
    inflated_env = inflate_tf_3d(env=env, res=res, radius_m=inflate_radius_m)
    indices = tf.stack([rows, cols, channels], axis=1)
    in_collision = tf.reduce_any(tf.gather_nd(inflated_env, indices) > occupied_threshold)
    return in_collision, inflated_env


def batch_in_collision_tf(environment: Dict,
                          xs,
                          ys,
                          inflate_radius_m: float,
                          occupied_threshold: float = 0.5):
    origin = environment['origin']
    res = environment['res']
    env = environment['env']
    gripper_rows, gripper_cols = batch_point_to_idx_tf(xs, ys, res, origin)
    inflated_env = inflate_tf(env=env, res=res, radius_m=inflate_radius_m)
    indices = tf.stack([gripper_rows, gripper_cols], axis=1)
    in_collision = tf.reduce_any(tf.gather_nd(inflated_env, indices) > occupied_threshold)
    return in_collision


def batch_in_collision_or_out_of_bounds_tf(environment: Dict,
                                           xs,
                                           ys,
                                           inflate_radius_m: float,
                                           occupied_threshold: float = 0.5):
    in_collision = batch_in_collision_tf(environment, xs, ys,
                                         inflate_radius_m=inflate_radius_m,
                                         occupied_threshold=occupied_threshold)
    out_of_bounds = batch_out_of_bounds_tf(environment, xs, ys)
    return tf.logical_or(in_collision, out_of_bounds)


def any_in_collision_or_out_of_bounds_tf(environment: Dict,
                                         xs,
                                         ys,
                                         inflate_radius_m: float,
                                         occupied_threshold: float = 0.5):
    in_collision_or_out_of_bounds = batch_in_collision_or_out_of_bounds_tf(environment, xs, ys,
                                                                           inflate_radius_m=inflate_radius_m,
                                                                           occupied_threshold=occupied_threshold)
    return tf.reduce_any(in_collision_or_out_of_bounds)


def gripper_interpolate_cc_and_oob(environment: Dict,
                                   xy0,
                                   xy1,
                                   inflate_radius_m: float,
                                   step_size_m: float = 0.01,
                                   occupied_threshold: float = 0.5):
    distance = np.linalg.norm(xy1 - xy0)
    steps = np.arange(0, distance + step_size_m, step_size_m)
    interpolated_points = np.expand_dims(xy0, axis=1) + np.outer((xy1 - xy0), steps)
    xs = interpolated_points[0]
    ys = interpolated_points[1]
    in_collision_or_out_of_bounds = batch_in_collision_or_out_of_bounds_tf(environment,
                                                                           xs=xs,
                                                                           ys=ys,
                                                                           inflate_radius_m=inflate_radius_m,
                                                                           occupied_threshold=occupied_threshold)
    return tf.reduce_any(in_collision_or_out_of_bounds)


def inflate_tf_3d(env, radius_m: float, res: float):
    h, w, c = env.shape
    radius = int(radius_m / res)
    s = 1 + 2 * radius
    conv = tf.keras.layers.Conv3D(filters=1,
                                  kernel_size=[s, s, s],
                                  padding='same',
                                  use_bias=False,
                                  weights=[tf.ones([s, s, s, 1, 1])])
    conv.build([1, h, w, c, 1])
    conv.set_weights([tf.ones([s, s, s, 1, 1])])
    x = tf.cast(env, tf.float32)[tf.newaxis, :, :, :, tf.newaxis]
    inflated = tf.squeeze(tf.clip_by_value(conv(x), clip_value_min=0, clip_value_max=1))
    return inflated


def inflate_tf(env, radius_m: float, res: float):
    h, w = env.shape
    radius = int(radius_m / res)
    s = 1 + 2 * radius
    conv = tf.keras.layers.Conv2D(filters=1,
                                  kernel_size=[s, s],
                                  padding='same',
                                  use_bias=False,
                                  weights=[tf.ones([s, s, 1, 1])])
    conv.build([1, h, w, 1])
    x = tf.cast(env, tf.float32)[tf.newaxis, :, :, tf.newaxis]
    inflated = tf.squeeze(tf.clip_by_value(conv(x), clip_value_min=0, clip_value_max=1))
    return inflated


def inflate(env: OccupancyData, radius_m: float, res: float):
    assert radius_m >= 0
    if radius_m == 0:
        return env

    inflated_data = np.copy(env)
    radius = int(radius_m / res)

    for i, j in np.ndindex(env.data.shape):
        try:
            if env.data[i, j] == 1:
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        r = i + di
                        c = j + dj
                        if 0 <= r < env.data.shape[0] and 0 <= c < env.data.shape[1]:
                            inflated_data[i + di, j + dj] = 1
        except IndexError:
            pass

    inflated = OccupancyData(data=inflated_data,
                             origin=env.origin,
                             resolution=env.resolution)
    return inflated
