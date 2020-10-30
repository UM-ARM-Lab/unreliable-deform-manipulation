from unittest import TestCase

import numpy as np

from moonshine.gpu_config import limit_gpu_mem
from moonshine.named_tensor import NamedTensor
from moonshine.named_tensor_utils import iter_nt

limit_gpu_mem(0.1)


class Test(TestCase):
    def test_named_tensor_named_indexing(self):
        x = np.arange(16).reshape([4, 2, 2])
        named_x = NamedTensor(x, ['batch', 'time'])

        np.testing.assert_almost_equal(named_x['batch', 0].data, np.array([[0, 1], [2, 3]]))
        self.assertEqual(named_x['batch', 0].dnames, ['time'])
        np.testing.assert_almost_equal(named_x['batch', 1].data, np.array([[4, 5], [6, 7]]))
        self.assertEqual(named_x['batch', 1].dnames, ['time'])

        np.testing.assert_almost_equal(named_x['time', 0][:, 0].data, np.array([0, 4, 8, 12]))
        self.assertEqual(named_x['time', 0].dnames, ['batch'])
        np.testing.assert_almost_equal(named_x['time', 1][0].data, np.array([2, 3]))
        self.assertEqual(named_x['time', 1].dnames, ['batch'])

    def test_named_tensor_named_indexing_oob(self):
        x = np.arange(16).reshape([4, 2, 2])
        named_x = NamedTensor(x, ['batch', 'time'])

        with self.assertRaises(IndexError) as _:
            named_x['batch', 10]

        with self.assertRaises(IndexError) as _:
            named_x['time', 10]

    def test_named_tensor_normal_indexing(self):
        x = np.arange(16).reshape([4, 2, 2])
        named_x = NamedTensor(x, ['batch', 'time'])

        np.testing.assert_almost_equal(named_x[0, 0, 0].data, 0)
        np.testing.assert_almost_equal(named_x[0, 0, 0].dnames, [])
        np.testing.assert_almost_equal(named_x[0, 0, 1].data, 1)
        np.testing.assert_almost_equal(named_x[0, 0, 1].dnames, [])
        np.testing.assert_almost_equal(named_x[0, 0].data, np.array([0, 1]))

        self.assertEqual(named_x[0].dnames, ['time'])
        self.assertEqual(named_x[:, 0].dnames, ['batch'])
        self.assertEqual(named_x[0, 0].dnames, [])

    def test_named_tensor_normal_indexing_oob(self):
        x = np.arange(16).reshape([4, 2, 2])
        named_x = NamedTensor(x, ['batch', 'time'])

        with self.assertRaises(IndexError) as _:
            named_x[10]

    def test_iter(self):
        x = np.arange(12).reshape([3, 2, 2])
        named_x = NamedTensor(x, ['batch', 'time'])
        y = np.arange(3).reshape([3, 1])
        named_y = NamedTensor(y, ['batch'])
        d = {
            'x': named_x,
            'y': named_y,
        }
        for d_t in iter_nt(d, 'time'):
            pass

        d_t_iter = iter(iter_nt(d, 'time'))
        d_t0 = next(d_t_iter)
        np.testing.assert_almost_equal(d_t0['x'].data, np.array([[0, 1], [4, 5], [8, 9]]))
        np.testing.assert_almost_equal(d_t0['y'].data, np.array([[0], [1], [2]]))
        d_t1 = next(d_t_iter)
        np.testing.assert_almost_equal(d_t1['x'].data, np.array([[2, 3], [6, 7], [10, 11]]))
        np.testing.assert_almost_equal(d_t1['y'].data, np.array([[0], [1], [2]]))
