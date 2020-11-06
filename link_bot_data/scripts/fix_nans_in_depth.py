#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama
import matplotlib.pyplot as plt
import numpy as np

import rospy
from link_bot_data.dynamics_dataset import DynamicsDatasetLoader
from link_bot_data.modify_dynamics_dataset import modify_dynamics_dataset
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.ros_pycommon import KINECT_MAX_DEPTH
from moonshine.moonshine_utils import numpify


def main():
    colorama.init(autoreset=True)
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=250, precision=5)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    rospy.init_node("fix_nans_in_dataset")

    outdir = args.dataset_dir.parent / (args.dataset_dir.name + '+nonan')

    def _process_example(dataset: DynamicsDatasetLoader, example: Dict):
        example = numpify(example)
        color_depth = example['color_depth_image']
        depth = color_depth[:, :, :, 3:4]
        color = color_depth[:, :, :, :3]
        new_depth = np.nan_to_num(depth, True, KINECT_MAX_DEPTH)
        new_color_depth = np.concatenate([color, new_depth], axis=-1)
        example['color_depth_image'] = new_color_depth
        yield example

    modify_dynamics_dataset(args.dataset_dir, outdir, _process_example)


if __name__ == '__main__':
    main()
