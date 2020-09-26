#!/usr/bin/env python

import argparse
import pathlib
import shutil
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import rospy
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature
from link_bot_pycommon.args import my_formatter
from moonshine.moonshine_utils import numpify


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=250, precision=5)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    rospy.init_node("filter_dataset")

    outdir = args.dataset_dir.parent / (args.dataset_dir.name + '+corrupt')
    outdir.mkdir(exist_ok=True, parents=True)

    record_options = tf.io.TFRecordOptions(compression_type='ZLIB')

    # load the dataset
    dataset = DynamicsDataset([args.dataset_dir])

    in_hparams = args.dataset_dir / 'hparams.json'
    out_hparams = outdir / 'hparams.json'
    shutil.copy(in_hparams, out_hparams)

    total_count = 0
    for mode in ['train', 'test', 'val']:
        tf_dataset = dataset.get_datasets(mode=mode)
        full_output_directory = outdir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        for i, example in enumerate(tf_dataset):
            print(mode, i)
            example = numpify(example)
            example = corrupt_example(dataset, example)

            # otherwise add it to the dataset
            features = {k: float_tensor_to_bytes_feature(v) for k, v in example.items()}
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))
            example = example_proto.SerializeToString()
            record_filename = "example_{:09d}.tfrecords".format(total_count)
            full_filename = full_output_directory / record_filename
            with tf.io.TFRecordWriter(str(full_filename), record_options) as writer:
                writer.write(example)
            total_count += 1


def corrupt_example(dataset: DynamicsDataset, example: Dict):
    k = 'link_bot'
    rope_points = example[k].reshape([dataset.sequence_length, -1, 3])

    gripper1_bias = np.random.uniform(-0.05, 0.05, size=3)
    gripper2_bias = np.random.uniform(-0.05, 0.05, size=3)
    example['gripper1'] += gripper1_bias
    example['gripper2'] += gripper2_bias

    mean = np.random.uniform(-0.05, 0.05)
    var = np.random.uniform(0.0, 0.02)
    rope_points += np.random.normal(mean, var, size=[dataset.sequence_length, 25, 3])
    r = np.random.rand()
    if r < 0.5:
        # zeros/missing/null data
        t = np.random.randint(0, dataset.sequence_length)
        rope_points[t] = 0
    elif r < 0.10:
        # reverse the points associations
        t = np.random.uniform(0, dataset.sequence_length)
        rope_points[t] = rope_points[t, ::-1, :]

    example['link_bot'] = rope_points.reshape([dataset.sequence_length, -1])
    return example


if __name__ == '__main__':
    main()
