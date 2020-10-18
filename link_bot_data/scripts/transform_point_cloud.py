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

from tf import transformations


def main():
    colorama.init(autoreset=True)
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=250, precision=5)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    rospy.init_node("transform_point_cloud")

    outdir = args.dataset_dir.parent / (args.dataset_dir.name + '+tfpc')

    def _process_example(dataset: DynamicsDataset, example: Dict):
        points = example.pop("cdcpd")
        points = np.reshape(points, [-1, 25, 3]).transpose([2, 0, 1]).reshape([3, -1])
        ones = np.ones([1, points.shape[1]])
        points_homogeneous = np.concatenate([points, ones], axis=0)
        transformation = transformations.euler_matrix(-1.721, 0.000, 1.571)
        transformation[0, 3] = 1.8
        transformation[1, 3] = 0.0
        transformation[2, 3] = 1.2
        transformed_points = transformation @ points_homogeneous
        transformed_points = transformed_points[:3].reshape([3, 20, 25]).transpose([1, 2, 0]).reshape([20, -1])
        example['cdcpd'] = transformed_points
        example.pop("joint_names")
        yield example

    modify_dynamics_dataset(args.dataset_dir, outdir, _process_example)


if __name__ == '__main__':
    main()
