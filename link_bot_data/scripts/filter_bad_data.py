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

    args = parser.parse_args()

    rospy.init_node("filter_dataset")

    outdir = args.dataset_dir.parent / (args.dataset_dir.name + '+filtered')
    record_options = tf.io.TFRecordOptions(compression_type='ZLIB')

    # load the dataset
    dataset = DynamicsDataset([args.dataset_dir])

    in_hparams = args.dataset_dir / 'hparams.json'
    out_hparams = outdir / 'hparams.json'
    shutil.copy(in_hparams, out_hparams)

    count = 0
    total_count = 0
    for mode in ['train', 'test', 'val']:
        tf_dataset = dataset.get_datasets(mode=mode)
        full_output_directory = outdir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        for i, example in enumerate(tf_dataset):
            example = numpify(example)
            rope_points = example['link_bot'].reshape([dataset.sequence_length, -1, 3])
            min_z_in_sequence = np.amin(np.amin(rope_points, axis=0), axis=0)[2]
            if min_z_in_sequence < 0.59:
                count += 1
                dataset.scenario.plot_environment_rviz(example)
                example_t = remove_batch(dataset.index_time(add_batch(example), 0))
                dataset.scenario.plot_state_rviz(example_t, label='')
                continue

            # otherwise add it to the dataset
            features = {k: float_tensor_to_bytes_feature(v) for k, v in example.items()}
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))
            example = example_proto.SerializeToString()
            record_filename = "example_{:09d}.tfrecords".format(total_count)
            full_filename = full_output_directory / record_filename
            with tf.io.TFRecordWriter(str(full_filename), record_options) as writer:
                writer.write(example)
            total_count += 1

    print(f"{count} / {total_count} removed")


if __name__ == '__main__':
    main()
