#!/usr/bin/env python

import argparse
import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

import rospy
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.modify_dynamics_dataset import modify_dynamics_dataset
from link_bot_pycommon.args import my_formatter


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=250, precision=5)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    rospy.init_node("filter_dataset")

    outdir = args.dataset_dir.parent / (args.dataset_dir.name + '+corrupt')
    outdir.mkdir(exist_ok=True, parents=True)

    modify_dynamics_dataset(args.dataset_dir, outdir, corrupt_example)


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
    yield example


if __name__ == '__main__':
    main()
