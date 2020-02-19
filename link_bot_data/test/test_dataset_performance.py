#!/usr/bin/env python
import argparse
import pathlib
import time
import random

import numpy as np
import tensorflow as tf

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset
from link_bot_pycommon.args import my_formatter

tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory', nargs='+')

    args = parser.parse_args()

    dataset = LinkBotStateSpaceDataset(args.dataset_dir)
    # params = {
    #     "pre_close_threshold": 0.1,
    #     "post_close_threshold": 0.1,
    #     "discard_pre_far": True,
    #     "balance": True,
    #     "type": "trajectory"
    # }
    # dataset = ClassifierDataset(args.dataset_dir, params)

    t0 = time.time()
    tf_dataset = dataset.get_datasets(mode='train', seed=1)
    time_to_load = time.time() - t0
    print("Time to Load (s): {:5.3f}".format(time_to_load))

    print("[NO CACHE] Time to Iterate (s):")
    for _ in range(4):
        t0 = time.time()
        for _ in tf_dataset.batch(64):
            pass
        time_to_iterate = time.time() - t0
        print("{:5.3f}".format(time_to_iterate))

    print("[CACHE] Time to Iterate (s):")
    tmpname = "/tmp/tf_{}".format(random.randint(0,100000))
    cached_tf_dataset = tf_dataset.cache(tmpname).batch(64)
    for _ in range(4):
        t0 = time.time()
        for _ in cached_tf_dataset:
            pass
        time_to_iterate = time.time() - t0
        print("{:5.3f}".format(time_to_iterate))


if __name__ == '__main__':
    main()
