#!/usr/bin/env python
import tensorflow as tf
import argparse
import pathlib

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_planning.visualization import plot_classifier_data

tf.enable_eager_execution()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=pathlib.Path)
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')

    args = parser.parse_args()

    classifier_dataset = ClassifierDataset(args.indir)
    dataset = classifier_dataset.get_dataset(mode=args.mode,
                                             shuffle=args.shuffle,
                                             num_epochs=1,
                                             seed=args.seed,
                                             compression_type=args.compression_type)

    for i, example_dict in enumerate(dataset):
        planned_sdf = example_dict['planned_sdf/sdf'].numpy()
        planned_sdf_extent = example_dict['planned_sdf/extent'].numpy()
        actual_sdf = example_dict['actual_sdf/sdf'].numpy()
        actual_sdf_extent = example_dict['actual_sdf/extent'].numpy()
        state = example_dict['state'].numpy()
        next_state = example_dict['next_state'].numpy()
        action = example_dict['action'].numpy()
        planned_state = example_dict['planned_state'].numpy()
        planned_next_state = example_dict['planned_next_state'].numpy()

        plot_classifier_data(actual_sdf, actual_sdf_extent, next_state, planned_next_state, planned_sdf, planned_sdf_extent,
                             planned_state, state, i)


if __name__ == '__main__':
    main()
