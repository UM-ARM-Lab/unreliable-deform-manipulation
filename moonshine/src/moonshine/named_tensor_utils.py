from typing import Dict

from moonshine.named_tensor import NamedTensor


def iter_nt(named_tensors: Dict[str, NamedTensor], dname: str):
    """
    generator for iterating over a dimension given its name
    """
    i = 0
    while True:
        v_i = {}
        for k, v in named_tensors.items():
            if dname not in v.dnames:
                v_i[k] = v
            else:
                try:
                    v_i[k] = v[dname, i]
                except IndexError:
                    return
        yield v_i
        i += 1


def add_batch(named_tensors: Dict[str, NamedTensor], name='batch'):
    """
    yields the elements along the dimension specified by dname
    """
    # calls add_dim on each tensor in the dictionary
    return {k: v.add_batch() for k, v in named_tensors.items()}


def remove_batch(named_tensors: Dict[str, NamedTensor], name='batch'):
    """
    yields the elements along the dimension specified by dname
    """
    # calls add_dim on each tensor in the dictionary
    return {k: v.add_batch() for k, v in named_tensors.items()}
