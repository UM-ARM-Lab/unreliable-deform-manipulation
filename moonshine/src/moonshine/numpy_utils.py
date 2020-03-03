from typing import Dict
import tensorflow as tf

import numpy as np


def remove_batch(*xs):
    if len(xs) == 1:
        return remove_batch_single(xs[0])
    else:
        return [remove_batch_single(x) for x in xs]


def add_batch(*xs):
    if len(xs) == 1:
        return add_batch_single(xs[0])
    else:
        return [add_batch_single(x) for x in xs]


def remove_batch_single(x):
    if isinstance(x, dict):
        return dict([(k, v[0]) for k, v in x.items()])
    else:
        return x[0]


def add_batch_single(x):
    if isinstance(x, np.ndarray):
        return np.expand_dims(x, axis=0)
    elif isinstance(x, tf.Tensor):
        return tf.expand_dims(x, axis=0)
    elif isinstance(x, dict):
        return dict([(k, add_batch_single(v)) for k, v in x.items()])
    else:
        return np.array([x])
