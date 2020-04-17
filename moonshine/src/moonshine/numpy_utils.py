import numpy as np
import tensorflow as tf


def listify(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, list):
        return [listify(x_i) for x_i in x]
    elif isinstance(x, tf.Tensor):
        x_np = x.numpy()
        return listify(x_np)
    elif isinstance(x, dict):
        return dict([(k, listify(v)) for k, v in x.items()])
    elif isinstance(x, float):
        return [x]
    else:
        raise NotImplementedError(type(x))


def states_are_equal(state_dict1, state_dict2):
    if state_dict1.keys() != state_dict2.keys():
        return False

    for key in state_dict1.keys():
        s1 = state_dict1[key]
        s2 = state_dict2[key]
        if not np.all(s1 == s2):
            return False

    return True


def dict_of_tensors_to_dict_of_numpy_arrays(tf_dict):
    np_dict = {}
    for k, v in tf_dict.items():
        np_dict[k] = v.numpy()
    return np_dict


def dict_of_sequences_to_sequence_of_dicts(dict_of_seqs):
    seq_of_dicts = []
    # assumes all values in the dict have the same first dimension size (num time steps)
    T = len(list(dict_of_seqs.values())[0])
    for t in range(T):
        dict_t = {}
        for k, v in dict_of_seqs.items():
            dict_t[k] = v[t]
        seq_of_dicts.append(dict_t)

    return seq_of_dicts


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
    elif isinstance(x, list) and isinstance(x[0], dict):
        return [(add_batch_single(v)) for v in x]
    elif isinstance(x, tf.Tensor):
        return tf.expand_dims(x, axis=0)
    elif isinstance(x, dict):
        return dict([(k, add_batch_single(v)) for k, v in x.items()])
    else:
        return np.array([x])
