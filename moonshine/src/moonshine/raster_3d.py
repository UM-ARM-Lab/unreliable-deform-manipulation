import numpy as np
import tensorflow as tf

from link_bot_pycommon.link_bot_sdf_utils import idx_to_point_3d


# TODO if meshgrid is slow, we can pull out the construction of the pixel coordinates (or memoize?)
def raster_3d(state, res, origin, h, w, c, k, batch_size: int):
    res = res[0]
    n_points = int(int(state.shape[1]) / 3)
    points = tf.reshape(state, [batch_size, n_points, 3])

    ## Below is a un-vectorized implementation, which is much easier to read and understand
    # rope_images = np.zeros([batch_size, h, w, c, n_points], dtype=np.float32)
    # for batch_index in range(batch_size):
    #     for point_idx in range(n_points):
    #         for row, col, channel in np.ndindex(h, w, c):
    #             point_in_meters = points[batch_index, point_idx]
    #             pixel_center_in_meters = idx_to_point_3d(row, col, channel, res, origin[batch_index])
    #             squared_distance = np.sum(np.square(point_in_meters - pixel_center_in_meters))
    #             pixel_value = np.exp(-k * squared_distance)
    #             rope_images[batch_index, row, col, channel, point_idx] += pixel_value

    points_y_x_z = tf.stack([points[:, :, 1], points[:, :, 0], points[:, :, 2]], axis=2)
    tiled_points_y_x_z = points_y_x_z[:, tf.newaxis, tf.newaxis, tf.newaxis]
    tiled_points_y_x_z = tf.tile(tiled_points_y_x_z, [1, h, w, c, 1, 1])
    # swap x and y
    pixel_row_indices = tf.range(0, h, dtype=tf.float32)
    pixel_col_indices = tf.range(0, w, dtype=tf.float32)
    pixel_channel_indices = tf.range(0, c, dtype=tf.float32)
    # pixel_indices is batch_size, n_points, 3
    pixel_indices = tf.stack(tf.meshgrid(pixel_row_indices, pixel_col_indices, pixel_channel_indices), axis=3)
    # add batch dim
    pixel_indices = tf.expand_dims(pixel_indices, axis=0)
    pixel_indices = tf.tile(pixel_indices, [batch_size, 1, 1, 1, 1])

    # shape [batch_size, h, w, c, 2]
    origin_expanded = origin[:, tf.newaxis, tf.newaxis, tf.newaxis]
    pixel_centers_y_x_z = (pixel_indices - origin_expanded) * res

    # add n_points dim
    pixel_centers_y_x_z = tf.expand_dims(pixel_centers_y_x_z, axis=4)
    pixel_centers_y_x_z = tf.tile(pixel_centers_y_x_z, [1, 1, 1, 1, n_points, 1])

    squared_distances = tf.reduce_sum(tf.square(pixel_centers_y_x_z - tiled_points_y_x_z), axis=5)
    pixel_values = tf.exp(-k * squared_distances)
    rope_images = tf.transpose(tf.reshape(pixel_values, [batch_size, h, w, c, n_points]), [0, 2, 1, 3, 4])
    return rope_images


def raster_3d_np(state, res, origin, h, w, c, k, batch_size: int):
    res = res[0]
    n_points = int(int(state.shape[1]) / 3)
    points = np.reshape(state, [batch_size, n_points, 3])

    ## Below is a un-vectorized implementation, which is much easier to read and understand
    rope_images = np.zeros([batch_size, h, w, c, n_points], dtype=np.float32)
    for batch_index in range(batch_size):
        for point_idx in range(n_points):
            for row, col, channel in np.ndindex(h, w, c):
                point_in_meters = points[batch_index, point_idx]
                pixel_center_in_meters = idx_to_point_3d(row, col, channel, res, origin[batch_index])
                squared_distance = np.sum(np.square(point_in_meters - pixel_center_in_meters))
                pixel_value = np.exp(-k * squared_distance)
                rope_images[batch_index, row, col, channel, point_idx] += pixel_value
    return rope_images
