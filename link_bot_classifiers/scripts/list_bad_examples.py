#!/usr/bin/env python
import argparse
import pathlib
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import rospy
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import add_predicted
from link_bot_pycommon.pycommon import print_dict
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import remove_batch, add_batch
from std_msgs.msg import Float32
from scipy import stats

limit_gpu_mem(1)


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')

    args = parser.parse_args()
    args.batch_size = 1

    rospy.init_node("visualize_classifier_data")

    classifier_dataset = ClassifierDataset(args.dataset_dirs)

    tf_dataset = classifier_dataset.get_datasets(mode='all')
    filenames = classifier_dataset.get_record_filenames(mode='all')
    iterator = iter(tf_dataset)
    i = 0
    s = None
    while True:
        try:
            example = next(iterator)
            s = example['env'].shape
        except StopIteration:
            break
        except tf.errors.InvalidArgumentError:
            bad_filename = filenames[i]
            print(bad_filename)
        i += 1
    print(s)

if __name__ == '__main__':
    main()
