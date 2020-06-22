import tensorflow as tf


def raster_3d(state, res, origin, h, w, c, k, batch_size: int):
    from link_bot_pycommon.link_bot_sdf_utils import idx_to_point_3d
    import numpy as np
    res = res[0]
    n_points = int(int(state.shape[1]) / 3)
    points = tf.reshape(state, [batch_size, n_points, 3])

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
