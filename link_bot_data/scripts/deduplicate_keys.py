#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama
import numpy as np

import rospy
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.modify_dynamics_dataset import modify_dynamics_dataset
from link_bot_pycommon.args import my_formatter
from moonshine.moonshine_utils import numpify


def main():
    colorama.init(autoreset=True)
    np.set_printoptions(suppress=True, linewidth=250, precision=5)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    rospy.init_node("transform_point_cloud")

    outdir = args.dataset_dir.parent / (args.dataset_dir.name + '+dedup')

    def _process_example(dataset: DynamicsDatasetLoader, example: Dict):
        example['rope'] = numpify(example['rope'])[::2]
        example['left_gripper'] = numpify(example['left_gripper'])[::2]
        example['right_gripper'] = numpify(example['right_gripper'])[::2]
        example.pop("joint_names")
        yield example

    modify_dynamics_dataset(args.dataset_dir, outdir, _process_example)


if __name__ == '__main__':
    main()
