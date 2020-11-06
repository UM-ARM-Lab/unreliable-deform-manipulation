#!/usr/bin/env python
import argparse
import pathlib
import pickle

import colorama
import matplotlib.pyplot as plt
import numpy as np
from progressbar import progressbar

from link_bot_data.recovery_dataset import RecoveryDatasetLoader
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)


def main():
    colorama.init(autoreset=True)

    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=5)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)

    args = parser.parse_args()

    outfilename = args.dataset_dir.parent / (args.dataset_dir.name + "_best")

    dataset = RecoveryDatasetLoader([args.dataset_dir])

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
