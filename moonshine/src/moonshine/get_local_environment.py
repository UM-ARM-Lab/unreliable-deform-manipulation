import numpy as np
import tensorflow as tf

from moonshine.moonshine_utils import add_batch
from moonshine.bilinear_sampler import bilinear_sampler


# @tf.function
def get_local_env_and_origin_np(center_point,
                                full_env,
                                full_env_origin,
                                res,
                                local_h_rows: int,
                                local_w_cols: int,
                                batch_size: int):
    """
    :param center_point: [batch, 2]
    :param full_env: [batch, h, w]
    :param full_env_origin: [batch, 2]
    :param res: [batch]
    :param local_h_rows: scalar
    :param local_w_cols: scalar
    :return:
    
    note from the TF docs:
     On CPU, if an out of bound index is found, an error is returned.
     On GPU, if an out of bound index is found, a 0 is stored in the corresponding output value.
    """
    center_cols = center_point[:, 0] / res + full_env_origin[:, 1]
    col_min = tf.cast(center_cols - local_w_cols / 2, tf.int64)
    col_max = tf.cast(center_cols + local_w_cols / 2, tf.int64)
    center_rows = center_point[:, 1] / res + full_env_origin[:, 0]
    row_min = tf.cast(center_rows - local_h_rows / 2, tf.int64)
    row_max = tf.cast(center_rows + local_h_rows / 2, tf.int64)

    row_indices = tf.range(row_min, row_max)
    col_indices = tf.range(col_min, col_max)
    batch_indices = tf.range(0, batch_size)
    local_env_pixel_row_indices = tf.range(0, local_h_rows, dtype=tf.float32)
    local_env_pixel_col_indices = tf.range(0, local_w_cols, dtype=tf.float32)
    y_indices, x_indices = tf.meshgrid(local_env_pixel_row_indices, local_env_pixel_col_indices)

    gather_indices = [batch_indices, row_indices, col_indices]
    print(gather_indices)
    local_env = tf.gather_nd(full_env, gather_indices)
    print(local_env.shape)

    center_point_coordinates = tf.stack([center_rows, center_cols], axis=1)
    local_center = tf.stack([local_h_rows / 2, local_w_cols / 2], axis=0)
    local_env_origin = full_env_origin - center_point_coordinates + local_center
    return local_env, local_env_origin


@tf.function
def get_local_env_and_origin_differentiable(center_point,
                                            full_env,
                                            full_env_origin,
                                            res,
                                            local_h_rows: int,
                                            local_w_cols: int):
    """
    :param center_point: [batch, 2]
    :param full_env: [batch, h, w]
    :param full_env_origin: [batch, 2]
    :param res: [batch]
    :param local_h_rows: scalar
    :param local_w_cols: scalar
    :return:
    """
    res = tf.convert_to_tensor(res, dtype=tf.float32)
    local_h_rows = tf.convert_to_tensor(local_h_rows, dtype=tf.float32)
    local_w_cols = tf.convert_to_tensor(local_w_cols, dtype=tf.float32)
    full_env_origin = tf.convert_to_tensor(full_env_origin, dtype=tf.float32)
    batch_size = int(full_env.shape[0])
    full_h_rows = int(full_env.shape[1])
    full_w_cols = int(full_env.shape[2])

    local_center = tf.stack([local_h_rows / 2, local_w_cols / 2], axis=0)
    full_center = tf.stack([full_h_rows / 2, full_w_cols / 2], axis=0)

    center_cols = center_point[:, 0] / res + full_env_origin[:, 1]
    center_rows = center_point[:, 1] / res + full_env_origin[:, 0]
    center_point_coordinates = tf.stack([center_rows, center_cols], axis=1)
    local_env_origin = full_env_origin - center_point_coordinates + local_center
    local_to_full_offset = tf.cast(full_center - local_env_origin, tf.int64)

    local_env_pixel_row_indices = tf.cast(tf.range(0, local_h_rows), dtype=tf.int64)
    local_env_pixel_col_indices = tf.cast(tf.range(0, local_w_cols), dtype=tf.int64)
    y_indices, x_indices = tf.meshgrid(local_env_pixel_row_indices, local_env_pixel_col_indices)
    # Add batch
    batch_y_indices = tf.tile(tf.expand_dims(y_indices, axis=0), [batch_size, 1, 1])
    batch_x_indices = tf.tile(tf.expand_dims(x_indices, axis=0), [batch_size, 1, 1])
    # Transform into coordinate of the full_env
    batch_y_indices_in_full_env_frame = batch_y_indices + local_to_full_offset[:, 0, tf.newaxis, tf.newaxis]
    batch_x_indices_in_full_env_frame = batch_x_indices + local_to_full_offset[:, 1, tf.newaxis, tf.newaxis]

    batch_indices = tf.tile(tf.range(0, batch_size, dtype=tf.int64)[:, tf.newaxis, tf.newaxis], [1, local_h_rows, local_w_cols])
    gather_indices = tf.stack([batch_indices, batch_y_indices_in_full_env_frame, batch_x_indices_in_full_env_frame], axis=3)
    image = tf.expand_dims(full_env, axis=3)
    local_image = tf.gather_nd(image, gather_indices)
    local_env = tf.transpose(local_image[:, :, :, 0], [0, 2, 1])

    return local_env, local_env_origin


def get_local_env_and_origin(center_point: np.ndarray,
                             full_env: np.ndarray,
                             full_env_origin: np.ndarray,
                             res: float,
                             local_h_rows: int,
                             local_w_cols: int):
    batched_inputs = add_batch(center_point, full_env, full_env_origin, np.float32(res))
    local_env, local_env_origin = get_local_env_and_origin_differentiable(*batched_inputs,
                                                                          local_h_rows=local_h_rows,
                                                                          local_w_cols=local_w_cols)

    # convert back from TF
    return local_env[0].numpy(), local_env_origin[0].numpy()
