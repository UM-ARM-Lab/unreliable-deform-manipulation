#!/usr/bin/env python

import argparse
import pathlib

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset
from link_bot_data.visualization import plot_rope_configuration, plottable_rope_configuration
from link_bot_pycommon.args import my_formatter

tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def plot_individual(train_dataset, redraw, states_description):
    for i, (input_data, output_data) in enumerate(train_dataset):
        # TODO: draw all kinds of states, not just link bot
        rope_configurations = input_data['state/link_bot'].numpy()
        actions = input_data['action'].numpy()

        full_env = input_data['full_env/env'].numpy()
        full_env_extents = input_data['full_env/extent'].numpy()

        fig, ax = plt.subplots()
        arrow_width = 0.1
        arena_size = 0.5

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim([-arena_size, arena_size])
        ax.set_ylim([-arena_size, arena_size])
        ax.axis("equal")

        full_env_handle = ax.imshow(np.flipud(full_env), extent=full_env_extents)
        head_point = rope_configurations[0].reshape(-1, 2)[-1]
        arrow = plt.Arrow(head_point[0], head_point[1], actions[0, 0], actions[0, 1], width=arrow_width, zorder=4)
        patch = ax.add_patch(arrow)
        link_bot_line = plot_rope_configuration(ax, rope_configurations[0], linewidth=1, zorder=3, c='r')[0]

        def update(t):
            nonlocal patch
            link_bot_config = rope_configurations[t]
            head_point = link_bot_config.reshape(-1, 2)[-1]
            action = actions[t]

            full_env_handle.set_data(np.flipud(full_env))
            full_env_handle.set_extent(full_env_extents)

            if redraw:
                xs, ys = plottable_rope_configuration(link_bot_config)
                link_bot_line.set_xdata(xs)
                link_bot_line.set_ydata(ys)
            else:
                plot_rope_configuration(ax, link_bot_config, linewidth=1, zorder=3, c='r')
            patch.remove()
            arrow = plt.Arrow(head_point[0], head_point[1], action[0], action[1], width=arrow_width, zorder=4)
            patch = ax.add_patch(arrow)

            ax.set_title("{} {}".format(i, t))

        interval = 50
        _ = FuncAnimation(fig, update, frames=actions.shape[0], interval=interval, repeat=True)
        plt.show()

        i += 1


def plot_all(train_dataset, states_description):
    """ Draws the first state from every trajectory in the dataset, assuming CONSTANT environment!!! """
    plt.figure()
    ax = plt.gca()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Training Dataset")
    ax.axis("equal")

    jet = cm.get_cmap("jet", 12)

    for i, (input_data, output_data) in enumerate(train_dataset):
        color_float_idx = (i % 100) / 100.0
        c = jet(color_float_idx)
        for state_key in states_description.keys():
            state_feature_name = 'state/{}'.format(state_key)
            states_traj = input_data[state_feature_name].numpy()
            first_state = states_traj[0]
            plot_rope_configuration(ax, first_state, linewidth=1, alpha=0.3, c=c, scatt=False)

    plt.savefig('dataset_visualization.png', transparent=True, dpi=600)
    plt.show()


def main():
    plt.style.use("paper")
    np.set_printoptions(suppress=True, linewidth=250, precision=3)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory', nargs='+')
    parser.add_argument('plot_type', choices=['individual', 'all'], default='individual')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='train', help='train test or val')
    parser.add_argument('--shuffle', action='store_true', help='shuffle')
    parser.add_argument('--redraw', action='store_true', help='redraw')

    args = parser.parse_args()

    np.random.seed(1)
    tf.random.set_random_seed(1)

    # load the dataset
    dataset = LinkBotStateSpaceDataset(args.dataset_dir)
    train_dataset = dataset.get_datasets(mode=args.mode,
                                         sequence_length=None,
                                         n_parallel_calls=1)
    if args.shuffle:
        train_dataset = train_dataset.shuffle(1024, seed=1)

    # print info about shapes
    input_data, output_data = next(iter(train_dataset))
    print("INPUTS:")
    for k, v in input_data.items():
        print(k, v.shape)
    print("OUTPUTS:")
    for k, v in output_data.items():
        print(k, v.shape)

    if args.plot_type == 'individual':
        plot_individual(train_dataset, args.redraw, dataset.states_description)
    elif args.plot_type == 'all':
        plot_all(train_dataset, dataset.states_description)


if __name__ == '__main__':
    main()
