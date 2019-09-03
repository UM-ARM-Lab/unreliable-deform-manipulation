#!/usr/bin/env python

import argparse

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Fore

from link_bot_data.visualization import plot_rope_configuration
from video_prediction.datasets import dataset_utils


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=1000)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('indir')
    parser.add_argument('dataset_hparams_dict')

    args = parser.parse_args()

    np.random.seed(0)
    tf.random.set_random_seed(0)

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.1))
    gpf.reset_default_session(config=config)
    sess = gpf.get_default_session()

    train_dataset, train_inputs, steps_per_epoch = dataset_utils.get_inputs(args.indir,
                                                                            'link_bot_video',
                                                                            args.dataset_hparams_dict,
                                                                            '',
                                                                            mode='train',
                                                                            epochs=1,
                                                                            seed=0,
                                                                            batch_size=1,
                                                                            shuffle=False)

    while True:
        try:
            data = sess.run(train_inputs)
        except tf.errors.OutOfRangeError:
            print(Fore.RED + "Dataset does not contain {} examples.".format(args.n_training_examples) + Fore.RESET)
            return

        rope_configurations = data['rope_configurations'].squeeze()
        actions = data['actions'].squeeze()
        sdf = np.flipud(np.transpose(data['sdf'].squeeze(), [0, 2, 1]))
        sdf = sdf > 0

        for config, action in zip(rope_configurations, actions):
            arrow_width = 0.02
            arena_size = 0.5

            ax = plt.gca()
            ax.clear()
            ax.imshow(sdf[0], extent=[-0.55, 0.55, -0.55, 0.55])

            plot_rope_configuration(ax, config, linewidth=5, zorder=1, c='r')
            arrow = plt.Arrow(config[4], config[5], action[0], action[1], width=arrow_width, zorder=2)
            ax.add_patch(arrow)

            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.xlim([-arena_size, arena_size])
            plt.ylim([-arena_size, arena_size])
            plt.axis("equal")

            plt.show()


if __name__ == '__main__':
    main()
