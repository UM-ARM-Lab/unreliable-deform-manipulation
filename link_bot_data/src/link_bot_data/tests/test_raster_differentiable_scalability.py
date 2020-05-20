#!/usr/bin/env python
from tabulate import tabulate
from time import perf_counter

import numpy as np
import tensorflow as tf

from moonshine.image_functions import raster_differentiable


def main():
    k = 1000
    h = 100
    w = 100
    n = 8192
    headers = ['batch size', f'time to raster {n} images (s)']
    table_data = []
    for pow in range(1, 12):
        batch_size = 2 ** pow

        res = tf.convert_to_tensor([0.1] * batch_size, dtype=tf.float32)
        origin = tf.convert_to_tensor([[100.0, 100.0]] * batch_size, dtype=tf.float32)
        state = tf.convert_to_tensor([np.random.randn(22)] * batch_size, dtype=tf.float32)
        # warmup
        raster_differentiable(state, res, origin, h, w, k, batch_size)

        t0 = perf_counter()
        for i in range(int(n / batch_size)):
            raster_differentiable(state, res, origin, h, w, k, batch_size)
        dt = perf_counter() - t0
        table_data.append([batch_size, dt])
    print(tabulate(table_data, headers=headers, tablefmt='fancygrid', floatfmt='.4f'))


if __name__ == '__main__':
    main()
