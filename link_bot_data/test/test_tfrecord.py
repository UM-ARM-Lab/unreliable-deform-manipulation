#!/usr/bin/env python3

import tensorflow as tf

from link_bot_data.link_bot_dataset_utils import bytes_feature

tf.compat.v1.enable_eager_execution()

import numpy as np
import argparse


def read(args):
    dataset = tf.data.TFRecordDataset([filename])

    feature_description = {
        'data': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse(example_proto):
        deserialized_dict = tf.io.parse_single_example(example_proto, feature_description)
        return tf.io.parse_tensor(deserialized_dict['data'], tf.float32)

    parsed_dataset = dataset.map(_parse)
    print(next(iter(parsed_dataset)))


def write(args):
    data = np.random.rand(10, 2)
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
    features = {
        'data': bytes_feature(tf.io.serialize_tensor(data_tensor).numpy()),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    example = example_proto.SerializeToString()
    writer = tf.io.TFRecordWriter(filename)
    writer.write(example)


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()
read_parser = subparsers.add_parser('read')
read_parser.set_defaults(func=read)
write_parser = subparsers.add_parser('write')
write_parser.set_defaults(func=write)

filename = '.tmp.tfrecords'

args = parser.parse_args()
args.func(args)
