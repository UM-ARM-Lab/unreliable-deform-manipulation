#!/usr/bin/env python

import pathlib
import argparse
import numpy as np
import tensorflow as tf

from link_bot_pycommon.args import my_formatter

tf.compat.v1.enable_eager_execution()

from link_bot_data.new_classifier_dataset import NewClassifierDataset
from link_bot_data.image_classifier_dataset import ImageClassifierDataset


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('example_idx', type=int, default=0)
    parser.add_argument('--dataset-type', choices=['new', 'image'], default='new')
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='train')

    args = parser.parse_args()

    if args.dataset_type == 'image':
        c = ImageClassifierDataset(args.dataset_dirs)
        dataset = c.get_datasets(mode=args.mode, shuffle=False, seed=0, batch_size=1)

        s0 = None
        for i, e in enumerate(dataset):
            s0 = e['image'].numpy()
            label = e['label'].numpy().squeeze()
            if i >= args.example_idx:
                print("label: {}".format(label))
                break

        count = 0
        for i, e in enumerate(dataset):
            s = e['image'].numpy()
            if np.allclose(s, s0):
                print(i)
                count += 1
        print("Count: {}".format(count))
    elif args.dataset_type == 'new':
        c = NewClassifierDataset(args.dataset_dirs)
        dataset = c.get_datasets(mode=args.mode, shuffle=False, seed=0, batch_size=1)

        s0 = None
        for i, e in enumerate(dataset):
            s0 = e['state'].numpy()
            label = e['label'].numpy().squeeze()
            if i >= args.example_idx:
                print("label: {}".format(label))
                break

        count = 0
        for i, e in enumerate(dataset):
            s = e['state'].numpy()
            if np.allclose(s, s0):
                print(i)
                count += 1
        print("Count: {}".format(count))
    else:
        raise ValueError('invalid dataset type')


if __name__ == '__main__':
    main()
