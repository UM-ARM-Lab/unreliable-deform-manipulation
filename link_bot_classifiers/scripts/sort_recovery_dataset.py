#!/usr/bin/env python
from progressbar import progressbar
import pickle
import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import pathlib
import rospy

from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import listify
from moonshine.gpu_config import limit_gpu_mem
from visualization_msgs.msg import MarkerArray, Marker
from link_bot_data.visualization import rviz_arrow
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import log_scale_0_to_1
from link_bot_data.recovery_dataset import RecoveryDataset


limit_gpu_mem(1)


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=5)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)

    args = parser.parse_args()

    outfilename = args.dataset_dir.parent / (args.dataset_dir.name + "_best")

    dataset = RecoveryDataset([args.dataset_dir])

    tf_dataset = dataset.get_datasets(mode='train')

    examples_to_sort = []
    for example in progressbar(tf_dataset):
        recovery_probability = example['recovery_probability'][1]
        if recovery_probability > 0.0:
            examples_to_sort.append(example)

    examples_to_sort = sorted(examples_to_sort, key=lambda e: e['recovery_probability'][1], reverse=True)

    print(f"saving {outfilename}")
    with outfilename.open("wb") as outfile:
        pickle.dump(examples_to_sort, outfile)


if __name__ == '__main__':
    main()
