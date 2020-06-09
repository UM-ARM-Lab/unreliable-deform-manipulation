import numpy as np
import tensorflow as tf
from time import perf_counter

from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_tensor_to_bytes_feature(value):
    return bytes_feature(tf.io.serialize_tensor(tf.convert_to_tensor(value, dtype=tf.float32)).numpy())


def parse_dataset(dataset, feature_description, n_parallel_calls=None):
    def _parse(example_proto):
        deserialized_dict = tf.io.parse_single_example(example_proto, feature_description)
        return deserialized_dict

    parsed_dataset = dataset.map(_parse, num_parallel_calls=n_parallel_calls)
    return parsed_dataset


y = np.random.rand(100, 100, 100, 10)
v = np.random.rand(2)

features = {
    'x': float_tensor_to_bytes_feature(v),
    'y': float_tensor_to_bytes_feature(y),
}

example_proto = tf.train.Example(features=tf.train.Features(feature=features))
example = example_proto.SerializeToString()
serialized_tensors = [example] * 10

dataset = tf.data.Dataset.from_tensor_slices(serialized_tensors)

fast_features_description = {
    'x': tf.io.FixedLenFeature([], tf.string),
}

parsed_dataset = parse_dataset(dataset, fast_features_description)

t0 = perf_counter()
for _ in parsed_dataset:
    pass
print(f'{perf_counter() - t0:.6f}')

slow_features_description = {
    'x': tf.io.FixedLenFeature([], tf.string),
    'y': tf.io.FixedLenFeature([], tf.string),
}

parsed_dataset = parse_dataset(dataset, slow_features_description)

t0 = perf_counter()
for _ in parsed_dataset:
    pass
print(f'{perf_counter() - t0:.6f}')
