#!/usr/bin/env python
import json
import argparse
import progressbar
import pathlib
import time
import random

import numpy as np
import tensorflow as tf

from link_bot_data.classifier_dataset import ClassifierDataset
import link_bot_classifiers 
from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset
from link_bot_data.link_bot_dataset_utils import balance, add_traj_image, cachename
from link_bot_pycommon.args import my_formatter

tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory', nargs='+')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='val')
    parser.add_argument('--n-repetitions', type=int, default=2)

    args = parser.parse_args()

    # dataset = LinkBotStateSpaceDataset(args.dataset_dir)
    params = {
        "pre_close_threshold": 0.1,
        "post_close_threshold": 0.1,
        "discard_pre_far": True,
        "balance": True,
        "type": "trajectory"
    }
    dataset = ClassifierDataset(args.dataset_dir, params)

    batch_size = 64

    t0 = time.perf_counter()
    tf_dataset = dataset.get_datasets(mode=args.mode)

    tf_dataset = tf_dataset.map(add_traj_image)
    tf_dataset = balance(tf_dataset, label_key='label')
    tf_dataset = tf_dataset.shuffle(1024)
    tf_dataset = tf_dataset.batch(batch_size)

    time_to_load = time.perf_counter() - t0
    print("Time to Load (s): {:5.3f}".format(time_to_load))

    # print("[NO CACHE] Time to Iterate (s):")
    # for _ in range(args.n_repetitions):
        # t0 = time.perf_counter()
        # for e in progressbar.progressbar(tf_dataset.batch(batch_size)):
            # pass
        # time_to_iterate = time.perf_counter() - t0

    # print("[CACHE] Time to Iterate (s):")
    try:
        for _ in range(args.n_repetitions):
            for e in progressbar.progressbar(tf_dataset):
                pass
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
