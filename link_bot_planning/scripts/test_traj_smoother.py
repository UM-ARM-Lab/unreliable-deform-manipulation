#!/usr/bin/env python
import argparse
import tensorflow as tf

from moonshine.raster_points_layer import differentiable_raster

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


def main():
    h = 50
    w = 50
    res = tf.constant([0.1], dtype=tf.float32, name='resolution')
    state = tf.constant([[0.19, 1.51, 2.12, 1.62, -0.01, 0.01]], dtype=tf.float32, name='state')
    origins = tf.constant([0.0, 0.0], dtype=tf.float32, name='origin')

    with tf.GradientTape(watch_accessed_variables=True) as tape:
        tape.watch(state)
        image = differentiable_raster(state, res, origins, h, w)

    print("Gradient:")
    print(tape.gradient(image, state))


if __name__ == '__main__':
    main()
