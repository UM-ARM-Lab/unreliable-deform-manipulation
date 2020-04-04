import numpy as np
import tensorflow as tf

from moonshine.numpy_utils import add_batch
from moonshine.bilinear_sampler import bilinear_sampler


# @tf.function
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
    b = int(full_env.shape[0])
    full_h_rows = int(full_env.shape[1])
    full_w_cols = int(full_env.shape[2])

    local_center = tf.stack([local_h_rows / 2, local_w_cols / 2], axis=0)
    full_center = tf.stack([full_h_rows / 2, full_w_cols / 2], axis=0)

    center_cols = center_point[:, 0] / res + full_env_origin[:, 1]
    center_rows = center_point[:, 1] / res + full_env_origin[:, 0]
    center_point_coordinates = tf.stack([center_rows, center_cols], axis=1)
    local_env_origin = full_env_origin - center_point_coordinates + local_center
    local_to_full_offset = full_center - local_env_origin

    local_env_pixel_row_indices = tf.range(0, local_h_rows, dtype=tf.float32)
    local_env_pixel_col_indices = tf.range(0, local_w_cols, dtype=tf.float32)
    y_indices, x_indices = tf.meshgrid(local_env_pixel_row_indices, local_env_pixel_col_indices)
    # Add batch
    batch_y_indices = tf.tile(tf.expand_dims(y_indices, axis=0), [b, 1, 1])
    batch_x_indices = tf.tile(tf.expand_dims(x_indices, axis=0), [b, 1, 1])
    # Transform into coordinate of the full_env
    batch_y_indices_in_full_env_frame = batch_y_indices + local_to_full_offset[:, 0, tf.newaxis, tf.newaxis]
    batch_x_indices_in_full_env_frame = batch_x_indices + local_to_full_offset[:, 1, tf.newaxis, tf.newaxis]

    image = tf.expand_dims(full_env, axis=3)
    # [b, h, w]
    local_image = bilinear_sampler(image, batch_x_indices_in_full_env_frame, batch_y_indices_in_full_env_frame)

    local_env = local_image[:, :, :, 0]

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
