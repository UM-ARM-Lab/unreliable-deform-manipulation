#!/usr/bin/env python
import argparse
import pathlib
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import progressbar

from link_bot_data.classifier_dataset import ClassifierDataset
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='train')
    parser.add_argument('--batch-size', type=int, default=4096)

    args = parser.parse_args()

    classifier_dataset = ClassifierDataset(args.dataset_dirs)
    dataset = classifier_dataset.get_datasets(mode=args.mode)

    dataset = dataset.batch(args.batch_size)

    positive_count = 0
    count = 0
    t0 = perf_counter()
    for example in progressbar.progressbar(dataset):
        is_close = example['is_close'].numpy().squeeze()
        positive_count += np.count_nonzero(is_close)
        count += is_close.size

        # Print statistics intermittently
        if count % 10 == 0:
            print_stats_and_timing(count, positive_count)

    total_dt = perf_counter() - t0

    print_stats_and_timing(count, positive_count, total_dt)


def print_stats_and_timing(count, positive_count, total_dt=None):
    print()  # newline for clarity
    negative_count = count - positive_count
    if total_dt is not None:
        print("Total iteration time = {:.4f}".format(total_dt))
    class_balance = positive_count / count * 100
    print("Number of examples: {}".format(count))
    print("Number positive: {}".format(positive_count))  # newline for clarity
    print("Number negative: {}".format(negative_count))
    print("Class balance: {:4.1f}% positive".format(class_balance))


if __name__ == '__main__':
    main()
