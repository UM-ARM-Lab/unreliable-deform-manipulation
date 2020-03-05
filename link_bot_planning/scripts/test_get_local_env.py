#!/usr/bin/env python

import tensorflow as tf

tf.compat.v1.enable_eager_execution()

from link_bot_pycommon.link_bot_sdf_utils import get_local_env_and_origin_differentiable, compute_extent
import numpy as np
import matplotlib.pyplot as plt

res = 0.05
full_h_rows = 100
full_w_cols = 150
local_h_rows = 30
local_w_cols = 30

center_point = np.array([[0.4, -.4]], np.float32)

full_env = np.zeros([1, full_h_rows, full_w_cols], dtype=np.float32)
full_env_origin = np.array([[full_h_rows / 2, full_w_cols / 2]], dtype=np.float32)
full_env[0, 50:71, 50:71] = 1.0
full_env[0, 10:31, 80:121] = 1.0

local_env, local_env_origin = get_local_env_and_origin_differentiable(center_point,
                                                                      full_env,
                                                                      full_env_origin,
                                                                      res,
                                                                      local_h_rows,
                                                                      local_w_cols)
local_extent = compute_extent(rows=local_h_rows, cols=local_w_cols, resolution=res, origin=local_env_origin[0])
full_extent = compute_extent(rows=full_h_rows, cols=full_w_cols, resolution=res, origin=full_env_origin[0])
plt.figure()
plt.imshow(np.flipud(full_env[0]), extent=full_extent, alpha=1.0)
plt.imshow(np.flipud(local_env[0]), extent=local_extent, alpha=0.3, cmap='Blues')
plt.xlim(full_extent[0:2])
plt.ylim(full_extent[2:4])
plt.show()
