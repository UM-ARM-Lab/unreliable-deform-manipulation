#!/usr/bin/env python

import argparse
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from matplotlib.animation import FuncAnimation

from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.visualization import plot_rope_configuration
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_planning.get_scenario import get_scenario
from link_bot_pycommon.args import my_formatter
from moonshine.image_functions import old_raster
from moonshine.moonshine_utils import add_batch


def plot_individual(train_dataset, scenario: ExperimentScenario, states_description):
    for i, (input_data, output_data) in enumerate(train_dataset):
        fig, ax = plt.subplots()

        actions = input_data['action'].numpy()
        full_env = input_data['full_env/env'].numpy()
        full_env_extents = input_data['full_env/extent'].numpy()

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(full_env_extents[0:2])
        ax.set_ylim(full_env_extents[2:4])
        ax.axis("equal")

        ax.imshow(np.flipud(full_env), extent=full_env_extents)

        first_state = {}
        for state_key in states_description.keys():
            states = input_data[state_key].numpy()
            first_state[state_key] = states[0]
        action_artist = scenario.plot_action(ax, first_state, actions[0], color='m', s=20, zorder=3)

        state_artist = scenario.plot_state(ax, first_state, color='b', s=10, zorder=2)

        def update(t):
            action_t = actions[t]
            state_t = {}
            for state_key in states_description.keys():
                state = input_data[state_key].numpy()[t]
                state_t[state_key] = state
            scenario.update_action_artist(action_artist, state_t, action_t)
            scenario.update_artist(state_artist, state_t)

            ax.set_title("{} {}".format(i, t))

        interval = 100
        anim = FuncAnimation(fig, update, frames=actions.shape[0], interval=interval, repeat=True)
        plt.show()

        i += 1


def plot_all(train_dataset, states_description):
    """ Draws states from a dataset, assuming CONSTANT environment!!! """
    plt.figure()
    ax = plt.gca()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Training Dataset")
    ax.axis("equal")

    jet = cm.get_cmap("jet", 12)

    for i, (input_data, output_data) in enumerate(train_dataset):
        full_env = input_data['full_env/env'].numpy().squeeze()
        full_env_extent = input_data['full_env/extent'].numpy().squeeze()
        plt.imshow(np.flipud(full_env), extent=full_env_extent, cmap='Greys')
        color_float_idx = (i % 10) / 10.0
        c = jet(color_float_idx)
        for state_key in states_description.keys():
            states_sequence = input_data[state_key].numpy()
            for t in range(0, states_sequence.shape[0], 1):
                state = states_sequence[t]
                plot_rope_configuration(ax, state, linewidth=4, alpha=0.7, c=c, scatt=False)

    plt.savefig('dataset_visualization.png', transparent=True, dpi=600)
    plt.show()


def plot_heatmap(train_dataset, show_env=True):
    # Get the environment stuff, which we assume is constant
    input_data, _ = next(iter(train_dataset))
    full_env = input_data['full_env/env'].numpy().squeeze()
    full_env_extent = input_data['full_env/extent'].numpy().squeeze()
    full_env_res = input_data['full_env/res'].numpy().squeeze()
    full_env_origin = input_data['full_env/origin'].numpy().squeeze()
    full_env_h, full_env_w = full_env.shape

    states_image = None
    states_image_mask = None
    for i, (input_data, output_data) in enumerate(train_dataset):
        if 'link_bot' in input_data.keys():
            states_sequence = input_data['link_bot'].numpy()
        elif 'gripper' in input_data.keys():
            states_sequence = input_data['gripper'].numpy()
        elif 'car' in input_data.keys():
            states_sequence = input_data['car'].numpy()
        else:
            raise ValueError('no supported state key was in the dataset.')
        for state in states_sequence:
            state_image_i = old_raster(*add_batch(state, full_env_res, full_env_origin), full_env_h, full_env_w)
            # merge down to one channel
            state_image_i = np.sum(state_image_i, axis=3, keepdims=True)
            state_image_i = state_image_i.astype(np.int32)
            if states_image_mask is None:
                states_image_mask = np.copy(state_image_i)
                states_image = np.copy(state_image_i)
            else:
                states_image_mask |= state_image_i
                states_image += state_image_i

    states_image = states_image.squeeze()
    states_image_mask = states_image_mask.squeeze()
    nonzero_indeces = np.nonzero(states_image)
    states_image_nonzero = states_image[nonzero_indeces]
    states_image_normalized = states_image.squeeze() / np.max(states_image)

    print('min', np.min(states_image_nonzero))
    print('max', np.max(states_image_nonzero))
    print('mean', np.mean(states_image_nonzero))
    print('median', np.median(states_image_nonzero))
    plt.figure()
    ax = plt.gca()
    ax.set_xlabel("Number of times occupied")
    ax.set_ylabel("count")
    ax.set_title("Training Dataset")
    plt.hist(states_image_nonzero, bins=100)

    plt.rcParams.update({'font.size': 22})
    plt.figure()
    ax = plt.gca()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Training Dataset")
    ax.axis("equal")

    full_env_mask = np.expand_dims(1 - full_env, axis=2)
    state_no_env_mask = full_env_mask * np.expand_dims(states_image_mask, axis=2)
    no_state_no_env_mask = full_env_mask * (1 - np.expand_dims(states_image_mask, axis=2))
    states_image_perceptually_uniform = cm.viridis(states_image_normalized)[:, :, :3]
    if show_env:
        states_image_masked = states_image_perceptually_uniform * state_no_env_mask
        full_env_inv = np.tile(full_env_mask, [1, 1, 3]) * no_state_no_env_mask
        combined_image = states_image_masked + full_env_inv
    else:
        combined_image = states_image_perceptually_uniform
    plt.imshow(np.flipud(combined_image), extent=full_env_extent)
    now = int(time.time())
    plt.savefig('dataset_visualization/{}.png'.format(now), dpi=600)
    plt.axis("equal")
    cb = plt.colorbar(ticks=[0, 1])
    cb.ax.set_yticklabels(["Least Occupied", "Most Occupied"])

    plt.show()


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=250, precision=3)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', type=pathlib.Path, help='dataset directory', nargs='+')
    parser.add_argument('plot_type', choices=['individual', 'all', 'heatmap', 'just_count'], default='individual')
    parser.add_argument('--take', type=int)
    parser.add_argument('--sequence-length', type=int, help='number of time steps per example')
    parser.add_argument('--mode', choices=['train', 'test', 'val', 'all'], default='train', help='train test or val')
    parser.add_argument('--shuffle', action='store_true', help='shuffle')
    parser.add_argument('--show-env', action='store_true', help='show env, assumed to be constant')

    args = parser.parse_args()

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
    input_data, output_data = next(iter(tf_dataset))
    print("INPUTS:")
    for k, v in input_data.items():
        print(k, v.shape)
    print("OUTPUTS:")
    for k, v in output_data.items():
        print(k, v.shape)

    scenario = get_scenario(dataset.hparams['scenario'])

    if args.plot_type == 'individual':
        plot_individual(tf_dataset, scenario, dataset.states_description)
    elif args.plot_type == 'all':
        plot_all(tf_dataset, dataset.states_description)
    elif args.plot_type == 'heatmap':
        plot_heatmap(tf_dataset, show_env=args.show_env)
    elif args.plot_type == 'just_count':
        i = 0
        for _ in tf_dataset:
            i += 1
        print(i)


if __name__ == '__main__':
    main()
