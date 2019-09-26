#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
import pathlib

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.visualization import plot_rope_configuration

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
        sdf = example_dict['sdf/sdf'].numpy()
        sdf_extent = example_dict['sdf/extent'].numpy()
        state = example_dict['state'].numpy()
        next_state = example_dict['next_state'].numpy()
        action = example_dict['action'].numpy()
        planned_state = example_dict['planned_state'].numpy()
        planned_next_state = example_dict['planned_next_state'].numpy()

        fig = plt.figure()
        axes = plt.subplot()

        plt.imshow(np.flipud(sdf.T) > 0, extent=sdf_extent, zorder=1)
        plot_rope_configuration(axes, state, c='red', label='state', zorder=2)
        plot_rope_configuration(axes, next_state, c='orange', label='next state', zorder=3)
        plot_rope_configuration(axes, planned_state, c='blue', label='planned state', zorder=4)
        plot_rope_configuration(axes, planned_next_state, c='cyan', label='planned next state', zorder=5)
        plt.axis("equal")
        plt.title("Example {}".format(i))
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
