#!/usr/bin/env python
import argparse
import pathlib
import time

import numpy as np
import tensorflow as tf

from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset
from link_bot_pycommon.args import my_formatter

tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory')
    parser.add_argument('--shuffle', action='store_true', help='shuffle')
    parser.add_argument('--sequence-length', type=int, default=10, help='sequence length. Must be < 100')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')

    args = parser.parse_args()

    dataset = LinkBotStateSpaceDataset(args.dataset_dir)

    t0 = time.time()
    tf_dataset = dataset.get_dataset(mode='train',
                                     shuffle=args.shuffle,
                                     seed=1,
                                     sequence_length=args.sequence_length,
                                     batch_size=args.batch_size)
    time_to_load = time.time() - t0
    print("Time to Load (s): {:5.3f}".format(time_to_load))

    print("Time to Iterate (s):")
    stats = []
    for _ in range(7):
        t0 = time.time()
        for _ in tf_dataset:
            pass
        time_to_iterate = time.time() - t0
        stats.append(time_to_iterate)
        print("{:5.3f}".format(time_to_iterate))

    print("{:5.3f}, {:5.3f}".format(np.mean(stats), np.std(stats)))


if __name__ == '__main__':
    main()
