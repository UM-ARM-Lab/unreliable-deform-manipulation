#!/usr/bin/env python
import argparse
import pathlib
from time import perf_counter
from colorama import Style

import matplotlib.pyplot as plt
import numpy as np
import progressbar

from link_bot_data.recovery_dataset import RecoveryDataset, is_stuck
from link_bot_data.link_bot_dataset_utils import num_reconverging, num_reconverging_subsequences
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)


def main():
    """
    1. remove transitions where the recovery probability at t=0 is positive, leaving only states where we are stuck.
    2. write a file containing the examples in sorted order
    """
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)

    args = parser.parse_args()

    dataset = RecoveryDataset([args.dataset_dir])
    for mode in ['train', 'test', 'val']:
        tf_dataset = dataset.get_datasets(mode=mode)
        tf_dataset = tf_dataset.filter(is_stuck)

        for example in progressbar.progressbar(tf_dataset):
            # write the example back out, and save some info for sorting


if __name__ == '__main__':
    main()
