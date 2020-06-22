#!/usr/bin/env python
import unittest

import numpy as np

from moonshine.gpu_config import limit_gpu_mem
from moonshine.raster_2d import raster_differentiable

limit_gpu_mem(0.1)


class Test(unittest.TestCase):

    def test_raster_differentiable1(self):
        state = np.array([[0, 0]], dtype=np.float32)
        res = [0.01]
        h = 5
        w = 5
        origin = np.array([[h // 2, w // 2]], dtype=np.float32)
        k = 100000
        batch_size = state.shape[0]

        image = raster_differentiable(state, res, origin, h, w, k, batch_size)

        self.assertAlmostEqual(image[0, 0, 0, 0].numpy(), 0)
        self.assertAlmostEqual(image[0, 1, 1, 0].numpy(), 0)
        self.assertAlmostEqual(image[0, 2, 2, 0].numpy(), 1)

    def test_raster_differentiable2(self):
        state = np.array([[-0.02, -0.02]], dtype=np.float32)
        res = [0.01]
        h = 5
        w = 5
        origin = np.array([[h // 2, w // 2]], dtype=np.float32)
        k = 100000
        batch_size = state.shape[0]

        image = raster_differentiable(state, res, origin, h, w, k, batch_size)

        self.assertAlmostEqual(image[0, 0, 0, 0].numpy(), 1)
        self.assertAlmostEqual(image[0, 1, 1, 0].numpy(), 0)
        self.assertAlmostEqual(image[0, 2, 2, 0].numpy(), 0)

    def test_raster_differentiable3(self):
        state = np.array([[-0.02, 0.02]], dtype=np.float32)
        res = [0.01]
        h = 5
        w = 5
        origin = np.array([[h // 2, w // 2]], dtype=np.float32)
        k = 100000
        batch_size = state.shape[0]

        image = raster_differentiable(state, res, origin, h, w, k, batch_size)

        self.assertAlmostEqual(image[0, 0, 0, 0].numpy(), 0)
        self.assertAlmostEqual(image[0, 1, 1, 0].numpy(), 0)
        self.assertAlmostEqual(image[0, 2, 2, 0].numpy(), 0)
        self.assertAlmostEqual(image[0, 4, 0, 0].numpy(), 1)
