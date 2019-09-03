#!/usr/bin/env python

import argparse

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation
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
                                                                            'sequence_length=100',
                                                                            mode='train',
                                                                            epochs=1,
                                                                            seed=0,
                                                                            batch_size=1,
                                                                            shuffle=False)

    i = 0
    while True:
        try:
            data = sess.run(train_inputs)
        except tf.errors.OutOfRangeError:
            print(Fore.RED + "Dataset does not contain {} examples.".format(args.n_training_examples) + Fore.RESET)
            return

        images = data['images'].squeeze()
        rope_configurations = data['rope_configurations'].squeeze()
        actions = data['actions'].squeeze()
        sdfs = np.transpose(data['sdf'].squeeze(), [0, 2, 1])
        sdfs = sdfs > 0

        fig, ax = plt.subplots()
        arrow_width = 0.02
        arena_size = 0.5

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim([-arena_size, arena_size])
        ax.set_ylim([-arena_size, arena_size])
        ax.axis("equal")

        img_handle = ax.imshow(images[0], extent=[-0.53, 0.53, -0.53, 0.53], zorder=1)
        sdf_img_handle = ax.imshow(np.flipud(sdfs[0]), extent=[-0.5, 0.5, -0.5, 0.5], alpha=0.5, zorder=2)

        arrow = plt.Arrow(rope_configurations[0, 4], rope_configurations[0, 5], actions[0, 0], actions[0, 1], width=arrow_width,
                          zorder=4)
        patch = ax.add_patch(arrow)

        def update(t):
            nonlocal patch
            image = images[t]
            config = rope_configurations[t]
            action = actions[t]
            sdf = sdfs[t]

            img_handle.set_data(image)
            sdf_img_handle.set_data(np.flipud(sdf))

            plot_rope_configuration(ax, config, linewidth=5, zorder=3, c='r')
            patch.remove()
            arrow = plt.Arrow(config[4], config[5], action[0], action[1], width=arrow_width, zorder=4)
            patch = ax.add_patch(arrow)

            ax.set_title("{} {}".format(i, t))

        anim = FuncAnimation(fig, update, frames=actions.shape[0], interval=1000, repeat=True)
        plt.show()

        i += 1


if __name__ == '__main__':
    main()
