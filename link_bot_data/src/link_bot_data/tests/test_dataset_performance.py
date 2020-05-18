#!/usr/bin/env python
import argparse
import pathlib
import os
import matplotlib.pyplot as plt
import psutil
from time import perf_counter

import progressbar

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_pycommon.args import my_formatter


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory', nargs='+')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='train')
    parser.add_argument('--n-repetitions', type=int, default=4)

    args = parser.parse_args()

    # dataset = DynamicsDataset(args.dataset_dir)
    dataset = ClassifierDataset(args.dataset_dir)

    t0 = perf_counter()
    tf_dataset = dataset.get_datasets(mode=args.mode)
    batch_size = 128
    tf_dataset = tf_dataset.batch(batch_size)

    tf_dataset = tf_dataset.shuffle(512)

    time_to_load = perf_counter() - t0
    print("Time to Load (s): {:5.3f}".format(time_to_load))

    try:
        ram_usage = []
        for _ in range(args.n_repetitions):
            t0 = perf_counter()
            for e in progressbar.progressbar(tf_dataset.take(800)):
                # print('{:.5f}'.format(perf_counter() - t0))
                process = psutil.Process(os.getpid())
                current_ram_usage = process.memory_info().vms
                ram_usage.append(current_ram_usage)
                pass
            print('{:.5f}'.format(perf_counter() - t0))
        plt.plot(ram_usage)
        plt.xlabel("iteration")
        plt.ylabel("ram usage (bytes)")
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
