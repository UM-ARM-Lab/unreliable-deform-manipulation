from typing import Dict

import numpy as np


def add_batch(*args):
    return [np.expand_dims(x, axis=0) for x in args]


def add_batch_to_dict(data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    new_dict = {}
    for k, v in data_dict.items():
        new_dict[k] = np.expand_dims(v, axis=0)
    return new_dict
