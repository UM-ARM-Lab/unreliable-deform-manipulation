from typing import Dict, List

import tensorflow as tf

from moonshine.moonshine_utils import numpify, remove_batch, add_batch


def index_batch_time(example: Dict, keys, b: int, t: int):
    e_t = {k: example[k][b, t] for k in keys}
    return e_t


def index_batch_time_with_metadata(metadata: Dict, example: Dict, keys, b: int, t: int):
    e_t = {k: example[k][b, t] for k in keys}
    e_t.update(metadata)
    return e_t


def index_time_with_metadata(metadata: Dict, example: Dict, keys, t: int):
    e_t = {k: example[k][t] for k in keys}
    e_t.update(metadata)
    return e_t


def index_time_np(e: Dict, time_indexed_keys: List[str], t: int):
    return numpify(index_time(e, time_indexed_keys, t))


def index_time(e: Dict, time_indexed_keys: List[str], t: int):
    return remove_batch(index_time_batched(add_batch(e), time_indexed_keys, t))


def index_time_batched(e: Dict, time_indexed_keys: List[str], t: int):
    e_t = {}
    for k, v in e.items():
        e_t[k] = index_time_batched_kv(e, k, t, time_indexed_keys, v)
    return e_t


def index_time_batched_kv(e, k, t, time_indexed_keys, v):
    if k in time_indexed_keys:
        if v.ndim == 1:
            return v
        elif t < v.shape[1]:
            return v[:, t]
        elif t == e[k].shape[1]:
            return v[:, t - 1]
        else:
            err_msg = f"time index {t} out of bounds for {k} which has shape {v.shape}"
            raise IndexError(err_msg)
    else:
        return v


def index_label_time_batched(example: Dict, t: int):
    if t == 0:
        # it makes no sense to have a label at t=0, labels are for transitions/sequences
        # the plotting function that consumes this should use None correctly
        return None
    return example['is_close'][:, t]


def index_dict_of_batched_tensors_np(in_dict: Dict, index: int, batch_axis: int = 0):
    out_dict_tf = index_dict_of_batched_tensors_tf(in_dict=in_dict, index=index, batch_axis=batch_axis)
    return {k: v.numpy() for k, v in out_dict_tf.items()}


def index_dict_of_batched_tensors_tf(in_dict: Dict, index: int, batch_axis: int = 0, keep_dims=False):
    out_dict = {}
    for k, v in in_dict.items():
        v_i = tf.gather(v, index, axis=batch_axis)
        if keep_dims:
            v_i = tf.expand_dims(v_i, axis=batch_axis)
        out_dict[k] = v_i
    return out_dict
