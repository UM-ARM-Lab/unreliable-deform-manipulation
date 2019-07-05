#!/usr/bin/env python
from __future__ import print_function

import argparse

import matplotlib.pyplot as plt
import numpy as np

import sdf_tools
from link_bot_data import random_environment_data_utils
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import link_bot_pycommon, link_bot_sdf_utils
from link_bot_pycommon import link_bot_sdf_tools
from link_bot_data.random_environment_data_utils import plot_sdf_and_ovs


def generate_env(args, env_idx):
    del env_idx
    sdf_resolution = np.array([args.res, args.res], dtype=np.float32)

    n_rows = int(args.h / args.res)
    n_cols = int(args.w / args.res)
    n_cells = n_rows * n_cols

    # pick random locations to place obstacles on a grid
    sdf_origin = np.array([n_rows // 2, n_cols // 2], dtype=np.int32)
    grid_world = np.zeros((n_rows, n_cols))
    occupied_cells = np.random.choice(n_cells, size=args.n_obstacles)
    occupied_cells_row, occupied_cells_col = np.unravel_index(occupied_cells, [n_rows, n_cols])
    for obstacle_row, obstacle_col in zip(occupied_cells_row, occupied_cells_col):
        for delta_row in range(-args.obstacle_size, args.obstacle_size + 1):
            for delta_col in range(-args.obstacle_size, args.obstacle_size + 1):
                r = (obstacle_row + int(delta_row)) % n_rows
                c = (obstacle_col + int(delta_col)) % n_cols
                grid_world[r, c] = 1

    # create a signed distance field from the grid
    sdf, sdf_gradient = sdf_tools.compute_2d_sdf_and_gradient(grid_world, args.res, sdf_origin)
    sdf_data = link_bot_sdf_tools.SDF(sdf, sdf_gradient, sdf_resolution, sdf_origin)

    # create random rope configurations
    rope_configurations = np.ndarray((args.steps, 6), dtype=np.float32)
    sdf_constraint_labels = np.ndarray((args.steps, 1), dtype=np.float32)
    ovs_constraint_labels = np.ndarray((args.steps, 1), dtype=np.float32)
    nominal_link_length = 0.5
    overstretched_threshold = nominal_link_length * args.overstretched_factor_threshold
    for i in range(args.steps):
        # half gaussian with variance such that ~50% of ropes will be overstretched
        length = abs(np.random.randn()) * 0.180 + nominal_link_length
        rope_configurations[i] = link_bot_pycommon.make_random_rope_configuration(sdf_data.extent, length=length)
        tail_x = rope_configurations[i, 0]
        tail_y = rope_configurations[i, 1]
        mid_x = rope_configurations[i, 2]
        mid_y = rope_configurations[i, 3]
        head_x = rope_configurations[i, 4]
        head_y = rope_configurations[i, 5]
        row, col = link_bot_sdf_utils.point_to_sdf_idx(head_x, head_y, sdf_resolution, sdf_origin)
        sdf_constraint_labels[i] = sdf[row, col] < args.sdf_threshold
        tail_mid_overstretched = np.hypot(tail_x - mid_x, tail_y - mid_y) > overstretched_threshold
        mid_head_overstretched = np.hypot(mid_x - head_x, mid_y - head_y) > overstretched_threshold
        ovs_constraint_labels[i] = tail_mid_overstretched or mid_head_overstretched

    all_labels = np.hstack((sdf_constraint_labels, ovs_constraint_labels))
    combined_constraint_labels = np.any(all_labels, axis=1)
    n_positive = np.count_nonzero(combined_constraint_labels)
    percentage_positive = n_positive * 100.0 / all_labels.shape[0]

    if args.n_plots and args.n_plots > 0:
        for i in np.random.choice(rope_configurations.shape[0], size=args.n_plots):
            plot_sdf_and_ovs(args, sdf_data, args.sdf_threshold, rope_configurations[i], sdf_constraint_labels[i],
                             ovs_constraint_labels[i])

        plt.show()

    labels_dict = {
        LabelType.SDF: sdf_constraint_labels,
        LabelType.Overstretching: ovs_constraint_labels,
        LabelType.Combined: combined_constraint_labels,
    }
    return rope_configurations, labels_dict, sdf_data, percentage_positive


def generate(args):
    full_output_directory = random_environment_data_utils.data_directory(args.outdir, args.envs, args.steps)

    if not args.seed:
        # I know this looks crazy, but the idea is that when we run the script multiple times we don't want to get the same output
        # but we als do want to be able to recreate the output from a seed, so we generate a random seed if non is provided
        args.seed = np.random.randint(0, 10000)
    np.random.seed(args.seed)

    save_dict_extras = {
        'sdf_threshold': args.sdf_threshold
    }
    random_environment_data_utils.generate_envs(args, full_output_directory, generate_env, save_dict_extras)


def plot_main(args):
    np.random.seed(args.seed)
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    generator = dataset.generator([], args.batch_size)
    xs, ys = generator[0]
    for i in range(args.n):
        sdf_data = link_bot_sdf_tools.SDF(sdf=np.squeeze(xs['sdf_input'][i]),
                                          gradient=xs['sdf_gradient'][i],
                                          resolution=xs['sdf_resolution'][i],
                                          origin=xs['sdf_origin'][i])
        rope_configuration = xs['rope_configuration'][i]
        if LabelType.SDF.name in ys and LabelType.Overstretching.name in ys:
            sdf_constraint_labels = ys[LabelType.SDF.name][i]
            ovs_constraint_labels = ys[LabelType.Overstretching.name][i]
            plot_sdf_and_ovs(args, sdf_data, 0, rope_configuration, sdf_constraint_labels, ovs_constraint_labels)
        else:
            combined_constraint_labels = ys[LabelType.Combined.name][i]
            plot_sdf_and_ovs(args, sdf_data, 0, rope_configuration, None, None, combined_constraint_labels)

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    generate_parser = subparsers.add_parser('generate')
    generate_parser.set_defaults(func=generate)
    generate_parser.add_argument('steps', type=int, help='number of data points per environment')
    generate_parser.add_argument('envs', type=int, help='number of environments')
    generate_parser.add_argument('w', type=int, help='environment with in meters (int)')
    generate_parser.add_argument('h', type=int, help='environment with in meters (int)')
    generate_parser.add_argument('--seed', type=int, help='random seed')
    generate_parser.add_argument('--res', '-r', type=float, default=0.05, help='size of cells in meters')
    generate_parser.add_argument('--n-obstacles', type=int, default=14, help='size of obstacles in cells')
    generate_parser.add_argument('--obstacle-size', type=int, default=7, help='size of obstacles in cells')
    generate_parser.add_argument('--sdf-threshold', type=np.float32, default=0.0)
    generate_parser.add_argument('--overstretched-factor-threshold', type=np.float32, default=1.25)
    generate_parser.add_argument('--n-plots', type=int, help='number of examples to plot')
    generate_parser.add_argument('--outdir')

    plot_parser = subparsers.add_parser('plot')
    plot_parser.set_defaults(func=plot_main)
    plot_parser.add_argument('dataset', help='json dataset file')
    plot_parser.add_argument('n', type=int, help='number of examples to plot')
    plot_parser.add_argument('--seed', type=int, help='random seed')
    plot_parser.add_argument('--batch-size', type=int, default=100)
    plot_parser.add_argument('--show-sdf-data', action='store_true')

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
