#!/usr/bin/env python

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from link_bot_data import link_bot_dataset_utils
from link_bot_data.visualization import plot_rope_configuration
from link_bot_pycommon.args import my_formatter

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=1000)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('input_dir', type=pathlib.Path, help='dataset directory')

    args = parser.parse_args()

    np.random.seed(0)
    tf.random.set_random_seed(0)

    dataset_hparams_dict = json.load(open(args.input_dir / 'hparams.json', 'r'))
    train_dataset, train_tf_dataset = link_bot_dataset_utils.get_dataset(args.input_dir,
                                                                         dataset_hparams_dict,
                                                                         'sequence_length=100',
                                                                         shuffle=False,
                                                                         mode='train',
                                                                         epochs=1,  # we handle epochs in our training loop
                                                                         seed=1,
                                                                         batch_size=1)

    i = 0
    for input_data, output_data in train_tf_dataset:
        rope_configurations = input_data['state'].numpy().squeeze()
        actions = input_data['actions'].numpy().squeeze()
        local_envs = input_data['actual_local_env/env'].numpy().squeeze()
        extents = input_data['actual_local_env/extent'].numpy().squeeze()

        fig, ax = plt.subplots()
        arrow_width = 0.02
        arena_size = 0.5

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim([-arena_size, arena_size])
        ax.set_ylim([-arena_size, arena_size])
        ax.axis("equal")

        local_env_handle = ax.imshow(np.flipud(local_envs[0]), extent=extents[0])

        arrow = plt.Arrow(rope_configurations[0, 4], rope_configurations[0, 5], actions[0, 0], actions[0, 1], width=arrow_width,
                          zorder=4)
        patch = ax.add_patch(arrow)

        def update(t):
            nonlocal patch
            config = rope_configurations[t]
            action = actions[t]
            local_env = local_envs[t]

            local_env_handle.set_data(np.flipud(local_env))
            local_env_handle.set_extent(extents[t])

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
