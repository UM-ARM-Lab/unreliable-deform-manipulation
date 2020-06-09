import tensorflow as tf


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


v = [1, 2, 3]

features = {
    'x': float_tensor_to_bytes_feature(v)
}

example_proto = tf.train.Example(features=tf.train.Features(feature=features))
example = example_proto.SerializeToString()
serialized_tensors = [example]

dataset = tf.data.Dataset.from_tensor_slices(serialized_tensors)

features_description = {
    'x': tf.io.FixedLenFeature([], tf.string),
}
parsed_dataset = parse_dataset(dataset, features_description)

print(next(iter(parsed_dataset)))
