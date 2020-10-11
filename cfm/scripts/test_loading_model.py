import argparse
import pathlib

import colorama
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import ros_numpy
import rospy
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_planning.shooting_method import ShootingMethod
from link_bot_planning.trajectory_optimizer import TrajectoryOptimizer
from link_bot_pycommon.floating_rope_scenario import publish_color_image
from link_bot_pycommon.rviz_animation_controller import RvizSimpleStepper
from moonshine.moonshine_utils import numpify, remove_batch, add_batch
from sensor_msgs.msg import Image
from state_space_dynamics import model_utils, filter_utils


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dirs", type=pathlib.Path, nargs="+")
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--mode", type=str, choices=['train', 'val', 'test'], default='val')

    args = parser.parse_args()

    rospy.init_node("test_load")

    test_dataset = DynamicsDataset(args.dataset_dirs)
    test_tf_dataset = test_dataset.get_datasets(mode=args.mode)

    for i in range(3):
        for example_idx, example in enumerate(test_tf_dataset):
            latent_dynamics_model, _ = model_utils.load_generic_model([args.checkpoint])
            for n in latent_dynamics_model.nets:
                print(n.get_weights()[0])

            filter_model = filter_utils.load_filter([args.checkpoint])
            for n in filter_model.nets:
                print(n.get_weights()[0])


if __name__ == '__main__':
    main()
