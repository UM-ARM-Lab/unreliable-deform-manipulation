#!/usr/bin/env python

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset
from link_bot_data.visualization import plot_rope_configuration
from link_bot_pycommon import link_bot_pycommon
from link_bot_pycommon.args import my_formatter

tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory', nargs='+')
    parser.add_argument('--no-plot', action='store_true', help='only print statistics')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='train', help='train test or val')
    parser.add_argument('--shuffle', action='store_true', help='shuffle')

    args = parser.parse_args()

    np.random.seed(1)
    tf.random.set_random_seed(1)

    dataset = LinkBotStateSpaceDataset(args.dataset_dir)
    train_dataset = dataset.get_datasets(shuffle=args.shuffle,
                                         mode=args.mode,
                                         seed=1,
                                         n_parallel_calls=1,
                                         batch_size=1)

    half_w = dataset.hparams['env_w'] / 2 - 0.01
    half_h = dataset.hparams['env_h'] / 2 - 0.01

    def oob(point):
        if point[0] <= -half_w or point[0] >= half_w:
            return True
        if point[1] <= -half_h or point[1] >= half_h:
            return True
        return False

    i = 0
    angles = []
    for input_data, output_data in train_dataset:

        out_of_bounds = False

        rope_configurations = input_data['state_s'].numpy().squeeze()
        actions = input_data['action_s'].numpy().squeeze()
        # local_envs = input_data['actual_local_env_s/env'].numpy().squeeze()
        # local_env_extents = input_data['actual_local_env_s/extent'].numpy().squeeze()
        full_env = input_data['full_env/env'].numpy().squeeze()
        full_env_extents = input_data['full_env/extent'].numpy().squeeze()

        for config in rope_configurations:
            state_angle = link_bot_pycommon.angle_from_configuration(config)
            angles.append(state_angle)

        if not args.no_plot:
            fig, ax = plt.subplots()
            arrow_width = 0.02
            arena_size = 0.5

            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_xlim([-arena_size, arena_size])
            ax.set_ylim([-arena_size, arena_size])
            ax.axis("equal")

            full_env_handle = ax.imshow(np.flipud(full_env), extent=full_env_extents)
            # local_env_handle = ax.imshow(np.flipud(local_envs[0]), extent=local_env_extents[0])

            arrow = plt.Arrow(rope_configurations[0, 4], rope_configurations[0, 5], actions[0, 0], actions[0, 1],
                              width=arrow_width, zorder=4)
            patch = ax.add_patch(arrow)

            def update(t):
                nonlocal patch, out_of_bounds
                config = rope_configurations[t]
                action = actions[t]
                # local_env = local_envs[t]

                # local_env_handle.set_data(np.flipud(local_env))
                # local_env_handle.set_extent(local_env_extents[t])

                full_env_handle.set_data(np.flipud(full_env))
                full_env_handle.set_extent(full_env_extents)

                plot_rope_configuration(ax, config, linewidth=5, zorder=3, c='r')
                patch.remove()
                arrow = plt.Arrow(config[4], config[5], action[0], action[1], width=arrow_width, zorder=4)
                patch = ax.add_patch(arrow)

                ax.set_title("{} {}".format(i, t))

            interval = 50
            _ = FuncAnimation(fig, update, frames=actions.shape[0], interval=interval, repeat=True)
            plt.show()

        for config in rope_configurations:
            p1 = config[0:2]
            p2 = config[2:4]
            p3 = config[4:6]
            out_of_bounds = out_of_bounds or oob(p1)
            out_of_bounds = out_of_bounds or oob(p2)
            out_of_bounds = out_of_bounds or oob(p3)
            if out_of_bounds:
                print("Example {} is goes of bounds!".format(i))
                break

        i += 1

    plt.figure()
    plt.hist(angles)
    plt.xlabel("angle (rad)")
    plt.ylabel("count")
    # plt.title("Hist for angle on dataset: {}".format(args.dataset_dir.name))
    plt.show()


if __name__ == '__main__':
    main()
