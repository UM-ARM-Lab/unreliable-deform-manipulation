#!/usr/bin/env python
import argparse
import pathlib

import colorama
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import rospy
from link_bot_data.classifier_dataset import ClassifierDataset
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)


def main():
    colorama.init(autoreset=True)
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
