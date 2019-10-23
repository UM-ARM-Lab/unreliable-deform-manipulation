import numpy as np


def add_batch(x, ndim_without_batch):
    """
    :param x: numpy array
    :param ndim_without_batch: number of dimensions EXCLUDING batch dimension that you want
    :return:
    """
    if isinstance(x, list):
        return np.expand_dims(x, 0)
    if x.ndim == ndim_without_batch + 1:
        return x
    elif x.ndim == ndim_without_batch:
        return np.expand_dims(x, 0)
    else:
        raise ValueError("x has {} dimensions but you asked to make it have {}".format(x.ndim, ndim_without_batch + 1))
