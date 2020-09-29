#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama
import matplotlib.pyplot as plt
import numpy as np

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
    parser.add_argument('rows', type=int, help='rows')
    parser.add_argument('cols', type=int, help='cols')
    parser.add_argument('channels', type=int, help='channels')

    args = parser.parse_args()

    rospy.init_node("resize_freespace_env")

    outdir = args.dataset_dir.parent / (args.dataset_dir.name + '+resized')

    def _process_example(dataset: DynamicsDataset, example: Dict):
        example['env'] = np.zeros([args.rows, args.cols, args.channels], dtype=np.float32)
        yield example

    modify_dynamics_dataset(args.dataset_dir, outdir, _process_example)


if __name__ == '__main__':
    main()
