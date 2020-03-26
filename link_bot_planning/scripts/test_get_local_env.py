#!/usr/bin/env python

import tensorflow as tf

from moonshine.image_functions import raster_differentiable

tf.compat.v1.enable_eager_execution()

from link_bot_pycommon.link_bot_sdf_utils import compute_extent
from moonshine.get_local_environment import get_local_env_and_origin_differentiable
import numpy as np
import matplotlib.pyplot as plt

res = [0.01, 0.01]
full_h_rows = 200
full_w_cols = 200
local_h_rows = 50
local_w_cols = 50

planned_state = np.array([[0.25, -0.3, 0.05, -0.52], [-0.5, -.05, -0.5, -0.1]], dtype=np.float32)
center_point = np.array([[planned_state[0, -2], planned_state[0, -1]],
                         [planned_state[1, -2], planned_state[1, -1]]], np.float32)

full_env = np.zeros([2, full_h_rows, full_w_cols], dtype=np.float32)
full_env_origin = np.array([[full_h_rows / 2, full_w_cols / 2], [full_h_rows / 2, full_w_cols / 2]], dtype=np.float32)
full_env[:, 50:71, 50:71] = 1.0
full_env[:, 10:31, 80:121] = 1.0

local_env, local_env_origin = get_local_env_and_origin_differentiable(center_point,
                                                                      full_env,
                                                                      full_env_origin,
                                                                      res,
                                                                      local_h_rows,
                                                                      local_w_cols)

rope_images = raster_differentiable(planned_state, res, local_env_origin, local_h_rows, local_w_cols)

for j in range(2):
    local_extent = compute_extent(rows=local_h_rows, cols=local_w_cols, resolution=res[j], origin=local_env_origin[j])
    full_extent = compute_extent(rows=full_h_rows, cols=full_w_cols, resolution=res[j], origin=full_env_origin[j])

    local_image = np.concatenate((np.expand_dims(local_env[j], axis=2), rope_images[j]), axis=2)

    plt.figure()
    plt.imshow(np.flipud(full_env[j]), extent=full_extent, alpha=1.0)
    plt.imshow(np.flipud(local_image), extent=local_extent, alpha=0.3)
    plt.scatter(center_point[j, 0], center_point[j, 1], s=2, marker='*')
    plt.xlim(full_extent[0:2])
    plt.ylim(full_extent[2:4])
    plt.show(block=True)
