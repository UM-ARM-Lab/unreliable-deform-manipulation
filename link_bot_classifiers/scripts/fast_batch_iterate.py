#!/usr/bin/env python
import argparse
import pathlib
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import progressbar

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import num_reconverging, num_reconverging_subsequences
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='train')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--take', type=int)

    args = parser.parse_args()

    classifier_dataset = ClassifierDataset(args.dataset_dirs)
    dataset = classifier_dataset.get_datasets(mode=args.mode, take=args.take)

    dataset = dataset.batch(args.batch_size)

    positive_count = 0
    count = 0
    reconverging_count = 0
    t0 = perf_counter()
    for example in progressbar.progressbar(dataset):
        is_close = example['is_close'].numpy().squeeze()
        positive_count += np.count_nonzero(is_close)
        reconverging_count += num_reconverging(is_close)
        count += is_close.size

        # Print statistics intermittently
        if count % 10 == 0:
            print_stats_and_timing(count, positive_count, reconverging_count)

    total_dt = perf_counter() - t0

    print_stats_and_timing(count, positive_count, reconverging_count, total_dt=total_dt)


def print_stats_and_timing(count, positive_count, reconverging_count, total_dt=None):
    print()  # newline for clarity
    if total_dt is not None:
        print(f"Total iteration time = {total_dt:.4f}")
    print(f"Total: {count}")
    print(f"Positive: {positive_count} ({positive_count / count * 100:3.2f}%)")
    print(f"Reconverging: {reconverging_count} ({reconverging_count / count * 100:3.2f}%)")


if __name__ == '__main__':
    main()
