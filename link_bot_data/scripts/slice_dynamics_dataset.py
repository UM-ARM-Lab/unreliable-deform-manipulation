#!/usr/bin/env python

import argparse
import json
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
    parser.add_argument('desired_sequence_length', type=int, help='desired seqeuence length')

    args = parser.parse_args()

    rospy.init_node("slice_dataset")

    outdir = args.dataset_dir.parent / (args.dataset_dir.name + f'+L{args.desired_sequence_length}')
    record_options = tf.io.TFRecordOptions(compression_type='ZLIB')

    # load the dataset
    dataset = DynamicsDataset([args.dataset_dir])

    in_hparams_filename = args.dataset_dir / 'hparams.json'
    out_hparams_filename = outdir / 'hparams.json'
    outdir.mkdir(exist_ok=True)
    with in_hparams_filename.open("r") as in_hparams_file:
        hparams = json.load(in_hparams_file)
    hparams['new_sequence_length'] = args.desired_sequence_length
    with out_hparams_filename.open("w") as out_hparams_file:
        json.dump(hparams, out_hparams_file)

    assert args.desired_sequence_length <= dataset.sequence_length

    count = 0
    total_count = 0
    for mode in ['train', 'test', 'val']:
        tf_dataset = dataset.get_datasets(mode=mode)
        full_output_directory = outdir / mode
        full_output_directory.mkdir(parents=True, exist_ok=True)

        for example in tf_dataset:
            # otherwise add it to the dataset
            out_examples = dataset.split_into_sequences(example, args.desired_sequence_length)
            for out_example in out_examples:
                out_example['time_idx'] = tf.range(0, args.desired_sequence_length, dtype=tf.float32)
                features = {k: float_tensor_to_bytes_feature(v) for k, v in out_example.items()}
                example_proto = tf.train.Example(features=tf.train.Features(feature=features))
                example = example_proto.SerializeToString()
                record_filename = "example_{:09d}.tfrecords".format(total_count)
                full_filename = full_output_directory / record_filename
                with tf.io.TFRecordWriter(str(full_filename), record_options) as writer:
                    writer.write(example)
                total_count += 1

            print(f"{total_count}")


if __name__ == '__main__':
    main()
