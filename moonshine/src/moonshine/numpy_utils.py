from typing import Dict
import tensorflow as tf

import numpy as np


def add_batch(*args):
    new_args = []
    for x in args:
        if isinstance(x, np.ndarray):
            new_args.append(np.expand_dims(x, axis=0))
        elif isinstance(x, tf.Tensor):
            new_args.append(tf.expand_dims(x, axis=0))
        elif isinstance(x, dict):
            new_args.append(add_batch_to_dict(x))
        else:
            new_args.append(np.array([x]))
    return new_args


def add_batch_to_dict(data_dict: Dict[str, np.ndarray]):
    new_dict = {}
    for k, v in data_dict.items():
        new_dict[k] = add_batch(v)
    return new_dict
