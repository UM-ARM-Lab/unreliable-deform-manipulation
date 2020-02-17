#!/usr/bin/env python3

import tensorflow as tf

from link_bot_data.link_bot_dataset_utils import bytes_feature, parse_and_deserialize

tf.compat.v1.enable_eager_execution()

import numpy as np
import argparse


def read(args):
    dataset = tf.data.TFRecordDataset([filename])

    feature_description = {
        'data': tf.io.FixedLenFeature([], tf.string)
    }

    deserialized_dataset = parse_and_deserialize(dataset, feature_description)

    print("deserialized example:")
    print(next(iter(deserialized_dataset)))


def write(args):
    data = np.random.rand(10, 2).astype(np.float32)
    print(data)
    features = {
        'data': bytes_feature(tf.io.serialize_tensor(data).numpy()),
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

np.random.seed(1)

args = parser.parse_args()
args.func(args)
