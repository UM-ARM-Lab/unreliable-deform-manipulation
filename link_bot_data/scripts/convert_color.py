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
from link_bot_pycommon.ros_pycommon import publish_color_image, publish_depth_image
from moonshine.moonshine_utils import numpify


def main():
    colorama.init(autoreset=True)
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=250, precision=5)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    rospy.init_node("convert_color")

    outdir = args.dataset_dir.parent / (args.dataset_dir.name + '+rgbd')

    def _process_example(dataset: DynamicsDatasetLoader, example: Dict):
        example = numpify(example)
        color_depth_image = example.pop('color_depth_image')
        b = color_depth_image[:, :, :, 0]
        g = color_depth_image[:, :, :, 1]
        r = color_depth_image[:, :, :, 2]
        d = color_depth_image[:, :, :, 3]
        rgbd = np.stack([r, g, b, d], axis=-1)
        publish_color_image(dataset.scenario.state_color_viz_pub, rgbd[0, :, :, :3])
        publish_depth_image(dataset.scenario.state_depth_viz_pub, rgbd[0, :, :, 3])
        example['rgbd'] = rgbd
        yield example

    modify_dynamics_dataset(args.dataset_dir, outdir, _process_example)


if __name__ == '__main__':
    main()
