#!/usr/bin/env python
import argparse
import pathlib
from typing import Dict

import colorama

import rospy
from link_bot_data.sort_dataset import sort_dataset
from link_bot_data.recovery_dataset import RecoveryDatasetLoader
from link_bot_pycommon.args import my_formatter


def main():
    colorama.init(autoreset=True)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    rospy.init_node("sort_dataset")

    dataset = RecoveryDatasetLoader([args.dataset_dir])

    def _get_value(dataset: RecoveryDatasetLoader, example: Dict):
        return float(example['recovery_probability'][1].numpy())

    sort_dataset(args.dataset_dir, dataset, _get_value, reverse=True)


if __name__ == '__main__':
    main()
