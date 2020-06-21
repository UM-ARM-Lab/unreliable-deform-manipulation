#!/usr/bin/env python

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import rospy
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_pycommon.animation_player import Player
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import numpify, add_batch, remove_batch


def plot_2d(dataset: DynamicsDataset, tf_dataset: tf.data.Dataset):
    for i, example in enumerate(tf_dataset):
        example = numpify(example)

        fig, ax = plt.subplots()

        actions = example['delta_position']
        environment = dataset.scenario.get_environment_from_example(example)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axis("equal")

        dataset.scenario.plot_environment(ax, environment)

        first_state = {}
        for state_key in dataset.state_feature_names:
            states = example[state_key]
            first_state[state_key] = states[0]
        action_artist = dataset.scenario.plot_action(ax, first_state, actions[0], color='m', s=20, zorder=3)

        state_artist = dataset.scenario.plot_state(ax, first_state, color='b', s=10, zorder=2)

        def update(t):
            action_t = actions[t]
            state_t = {}
            for _state_key in dataset.state_feature_names:
                state = example[_state_key][t]
                state_t[_state_key] = state
            dataset.scenario.update_action_artist(action_artist, state_t, action_t)
            dataset.scenario.update_artist(state_artist, state_t)

            ax.set_title("{} {}".format(i, t))

        interval = 100
        anim = Player(fig, update, max_index=actions.shape[0], interval=interval, repeat=True)
        plt.show()

        i += 1


def plot_3d(dataset: DynamicsDataset, tf_dataset: tf.data.Dataset):
    rospy.loginfo("Don't forget to start the viz_stepper")
    for i, example in enumerate(tf_dataset):
        example = numpify(example)

        dataset.scenario.plot_environment_rviz(example)

        time_steps = example['time_idx']
        anim = RvizAnimationController(time_steps)

        while not anim.done:
            t = anim.t()
            example_t = remove_batch(dataset.index_time(add_batch(example), t))
            dataset.scenario.plot_state_rviz(example_t, label='')
            dataset.scenario.plot_action_rviz(example_t, label='')

            # this will return when either the animation is "playing" or because the user stepped forward
            anim.step()


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=250, precision=5)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory', nargs='+')
    parser.add_argument('plot_type', choices=['2d', '3d', 'just_count'], default='2d')
    parser.add_argument('--take', type=int)
    parser.add_argument('--sequence-length', type=int, help='number of time steps per example')
    parser.add_argument('--mode', choices=['train', 'test', 'val', 'all'], default='train', help='train test or val')
    parser.add_argument('--shuffle', action='store_true', help='shuffle')
    parser.add_argument('--show-env', action='store_true', help='show env, assumed to be constant')

    args = parser.parse_args()

    rospy.init_node("visualize_dynamics_dataset")

    np.random.seed(1)
    tf.random.set_seed(1)

    # load the dataset
    dataset = DynamicsDataset(args.dataset_dir)
    tf_dataset = dataset.get_datasets(mode=args.mode,
                                      sequence_length=args.sequence_length,
                                      n_parallel_calls=1,
                                      take=args.take)

    if args.shuffle:
        tf_dataset = tf_dataset.shuffle(1024, seed=1)

    # print info about shapes
    example = next(iter(tf_dataset))
    print("Example:")
    for k, v in example.items():
        print(k, v.shape)

    if args.plot_type == '2d':
        plot_2d(dataset, tf_dataset)
    elif args.plot_type == '3d':
        # uses rviz
        plot_3d(dataset, tf_dataset)
    elif args.plot_type == 'just_count':
        i = 0
        for _ in tf_dataset:
            i += 1
        print(f'num examples {i}')


if __name__ == '__main__':
    main()
