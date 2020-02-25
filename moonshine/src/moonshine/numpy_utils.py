from typing import Dict
import tensorflow as tf

import numpy as np


def add_batch(*args):
    new_args = []
    for x in args:
        if isinstance(x, np.ndarray):
            new_args.append(np.expand_dims(x, axis=0))
        else:
            new_args.append(tf.expand_dims(x, axis=0))
    return new_args


def add_batch_to_dict(data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    new_dict = {}
    for k, v in data_dict.items():
        # TODO: handle numpy or tf
        new_dict[k] = np.expand_dims(v, axis=0)
    return new_dict
