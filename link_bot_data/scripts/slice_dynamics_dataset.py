#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import rospy
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.modify_dynamics_dataset import modify_dynamics_dataset
from link_bot_pycommon.args import my_formatter


def main():
    colorama.init(autoreset=True)
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=250, precision=5)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('desired_sequence_length', type=int, help='desired seqeuence length')

    args = parser.parse_args()

    rospy.init_node("slice_dataset")

    outdir = args.dataset_dir.parent / (args.dataset_dir.name + f'+L{args.desired_sequence_length}')

    def _process_example(dataset: DynamicsDataset, example: Dict):
        out_examples = dataset.split_into_sequences(example, args.desired_sequence_length)
        for out_example in out_examples:
            out_example['time_idx'] = tf.range(0, args.desired_sequence_length, dtype=tf.float32)
            yield out_example

    modify_dynamics_dataset(args.dataset_dir, outdir, _process_example)


if __name__ == '__main__':
    main()
