#!/usr/bin/env python
from time import time

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

N = 5000
W = 100
H = 100
C = 24

numpy_data = np.random.rand(N, W, H, C)
dataset = tf.data.Dataset.from_tensor_slices(numpy_data)

print("numpy")
np_dts = []
for d in dataset:
    t0 = time()
    _ = d.numpy().flatten()
    dt = time() - t0
    # np_dts.append(dt)
    # print(dt)
print("Mean: {:0.4f}".format(np.mean(np_dts)))

print("tf")
tf_dts = []
for d in dataset:
    t0 = time()
    _ = tf.reshape(d, [-1])
    dt = time() - t0
    tf_dts.append(dt)
    # print(dt)
print("Mean: {:0.4f}".format(np.mean(tf_dts)))
