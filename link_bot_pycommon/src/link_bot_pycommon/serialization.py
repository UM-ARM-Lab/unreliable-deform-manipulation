import gzip
import json
import pathlib
import uuid
from enum import Enum

import hjson
import numpy as np
import tensorflow as tf
from dataclasses_json import DataClassJsonMixin

from rospy_message_converter import message_converter
from sensor_msgs.msg import genpy


class MyHjsonEncoder(hjson.HjsonEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pathlib.Path):
            return obj.as_posix()
        elif isinstance(obj, DataClassJsonMixin):
            return obj.to_dict()
        elif np.isscalar(obj):
            return obj.item()
        elif isinstance(obj, Enum):
            return str(obj)
        elif isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, genpy.Message):
            return message_converter.convert_ros_message_to_dictionary(obj)
        return hjson.HjsonEncoder.default(self, obj)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pathlib.Path):
            return obj.as_posix()
        elif isinstance(obj, DataClassJsonMixin):
            return obj.to_dict()
        elif np.isscalar(obj):
            return obj.item()
        elif isinstance(obj, Enum):
            return str(obj)
        elif isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def my_dump(data, fp, indent=None):
    return json.dump(data, fp, cls=MyEncoder, indent=indent)


def my_hdump(data, fp, indent=None):
    return hjson.dump(data, fp, cls=MyHjsonEncoder)


def my_dumps(data):
    return json.dumps(data, cls=MyEncoder)


def my_hdumps(data):
    return hjson.dumps(data, cls=MyHjsonEncoder)


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


class MyHJsonSerializer:

    @staticmethod
    def dump(data, fp):
        hjson.dump(data, fp, cls=MyHjsonEncoder)
