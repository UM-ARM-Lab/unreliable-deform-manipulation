import tensorflow as tf
import numpy as np


def assert_dicts_close_np(a, b):
    for v1, v2 in zip(a.values(), b.values()):
        assert np.allclose(v1, v2)


def assert_dicts_close_tf(a, b, rtol=1e-4, atol=1e-8):
    for v1, v2 in zip(a.values(), b.values()):
        assert tf.reduce_all(tf.abs(v1 - v2) <= tf.abs(v2) * rtol + atol)


def assert_list_of_dicts_close_np(a, b):
    for a_i, b_i in zip(a, b):
        assert_dicts_close_np(a_i, b_i)


def assert_list_of_dicts_close_tf(a, b, rtol=1e-4, atol=1e-8):
    for a_i, b_i in zip(a, b):
        assert_dicts_close_tf(a_i, b_i, rtol, atol)