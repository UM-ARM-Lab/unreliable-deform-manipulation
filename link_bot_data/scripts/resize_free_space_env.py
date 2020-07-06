#!/usr/bin/env python

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import shutil
import tensorflow as tf

import rospy
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_pycommon.animation_player import Player
from link_bot_pycommon.args import my_formatter
from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import numpify, add_batch, remove_batch


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=250, precision=5)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('rows', type=int, help='rows')
    parser.add_argument('cols', type=int, help='cols')
    parser.add_argument('channels', type=int, help='channels')

    args = parser.parse_args()

    rospy.init_node("resize_freespace_env")

    outdir = args.dataset_dir.parent / (args.dataset_dir.name + '+resized')
    record_options = tf.io.TFRecordOptions(compression_type='ZLIB')

    # load the dataset
    dataset = DynamicsDataset([args.dataset_dir])

    in_hparams = args.dataset_dir / 'hparams.json'
    out_hparams = outdir / 'hparams.json'
    outdir.mkdir(exist_ok=True)
    shutil.copy(in_hparams, out_hparams)

    total_count = 0
    for mode in ['train', 'test', 'val']:
        tf_dataset = dataset.get_datasets(mode=mode)
        full_output_directory = outdir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        for i, example in enumerate(tf_dataset):
            example['env'] = np.zeros([args.rows, args.cols, args.channels], dtype=np.float32)

            # otherwise add it to the dataset
            features = {k: float_tensor_to_bytes_feature(v) for k, v in example.items()}
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))
            example = example_proto.SerializeToString()
            record_filename = "example_{:09d}.tfrecords".format(total_count)
            full_filename = full_output_directory / record_filename
            with tf.io.TFRecordWriter(str(full_filename), record_options) as writer:
                writer.write(example)
            total_count += 1


if __name__ == '__main__':
    main()
