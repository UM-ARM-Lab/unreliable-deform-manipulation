#!/usr/bin/env python
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from moonshine.image_functions import raster_differentiable_faster, raster_differentiable

state = np.array([[0, 0]], dtype=np.float32)
res = [0.03]
h = 200
w = 200
origin = np.array([[h/2, w/2]], dtype=np.float32)
k = 10000
batch_size = state.shape[0]

# call once
t0 = perf_counter()
for _ in range(10000):
    batched_images = raster_differentiable(state, res, origin, h, w, k, batch_size)
    # raster_differentiable_faster(state, res, origin, h, w, k, batch_size)
dt = perf_counter() - t0
print("{:.6f}".format(dt))

for image in batched_images:
    plt.imshow(np.flipud(image.numpy().squeeze()))
    plt.show()
