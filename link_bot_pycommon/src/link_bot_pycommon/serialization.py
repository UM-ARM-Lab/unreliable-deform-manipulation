import gzip
import numpy as np
import json

from enum import Enum
import tensorflow as tf
from dataclasses import dataclass
from dataclasses_json import dataclass_json, DataClassJsonMixin


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, DataClassJsonMixin):
            return obj.to_dict()
        elif np.isscalar(obj):
            return obj.item()
        elif isinstance(obj, Enum):
            return str(obj)
        elif isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        return json.JSONEncoder.default(self, obj)


def my_dump(data, fp):
    return json.dump(data, fp, cls=MyEncoder)


def my_dumps(data):
    return json.dumps(data, cls=MyEncoder)


def dummy_proof_write(data, filename):
    """
    takes your data, serializes it to json, then gzips the result.
    works with pretty complicated "data", including arbitrarily nested numpy and tensorflow types,
    as well as Enums and things decorated with dataclass_json. See MyEncoder for full support info
    """
    while True:
        try:
            with gzip.open(filename, 'wb') as data_file:
                data_str = my_dumps(data)
                data_file.write(data_str.encode("utf-8"))
            return
        except KeyboardInterrupt:
            pass
