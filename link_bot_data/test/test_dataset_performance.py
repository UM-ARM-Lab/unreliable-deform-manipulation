#!/usr/bin/env python
import argparse
import pathlib
import time

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
    parser.add_argument('--shuffle', action='store_true', help='shuffle')

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

    batch_size = 64

    t0 = time.time()
    tf_dataset = dataset.get_datasets(mode='train',
                                      shuffle=args.shuffle,
                                      seed=1,
                                      batch_size=batch_size)
    time_to_load = time.time() - t0
    print("Time to Load (s): {:5.3f}".format(time_to_load))

    print("Time to Iterate (s):")
    stats = []
    cached_tf_dataset = tf_dataset.cache('/tmp/test_perf_cache')
    for _ in range(5):
        t0 = time.time()
        for _ in cached_tf_dataset:
            pass
        time_to_iterate = time.time() - t0
        stats.append(time_to_iterate)
        print("{:5.3f}".format(time_to_iterate))

    print("{:5.3f}, {:5.3f}".format(np.mean(stats), np.std(stats)))

    print("Time to Iterate (s):")
    stats = []
    for _ in range(5):
        t0 = time.time()
        for _ in tf_dataset:
            pass
        time_to_iterate = time.time() - t0
        stats.append(time_to_iterate)
        print("{:5.3f}".format(time_to_iterate))

    print("{:5.3f}, {:5.3f}".format(np.mean(stats), np.std(stats)))


if __name__ == '__main__':
    main()
