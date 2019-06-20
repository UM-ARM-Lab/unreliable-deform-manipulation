#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore

import sdf_tools
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import link_bot_pycommon
from link_bot_pycommon.link_bot_pycommon import SDF


def plot(args, sdf_data, threshold, rope_configuration, constraint_labels):
    del args  # unused
    plt.figure()
    binary = sdf_data.sdf < threshold
    plt.imshow(np.flipud(binary.T), extent=sdf_data.extent)

    xs = [rope_configuration[0], rope_configuration[2], rope_configuration[4]]
    ys = [rope_configuration[1], rope_configuration[3], rope_configuration[5]]
    sdf_constraint_color = 'r' if constraint_labels[0] else 'g'
    overstretched_constraint_color = 'r' if constraint_labels[1] else 'g'
    plt.plot(xs, ys, linewidth=0.5, zorder=1, c=overstretched_constraint_color)
    plt.scatter(rope_configuration[4], rope_configuration[5], s=16, c=sdf_constraint_color, zorder=2)

    plt.figure()
    plt.imshow(np.flipud(sdf_data.sdf.T), extent=sdf_data.extent)
    subsample = 2
    x_range = np.arange(sdf_data.extent[0], sdf_data.extent[1], subsample * sdf_data.resolution[0])
    y_range = np.arange(sdf_data.extent[0], sdf_data.extent[1], subsample * sdf_data.resolution[1])
    y, x = np.meshgrid(y_range, x_range)
    dx = sdf_data.gradient[::subsample, ::subsample, 0]
    dy = sdf_data.gradient[::subsample, ::subsample, 1]
    plt.quiver(x, y, dx, dy, units='x', scale=10)


def generate_env(args):
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
    sdf_data = link_bot_pycommon.SDF(sdf, sdf_gradient, sdf_resolution, sdf_origin)

    # create random rope configurations
    rope_configurations = np.ndarray((args.n, 6), dtype=np.float32)
    constraint_labels = np.ndarray((args.n, 2), dtype=np.float32)
    nominal_link_length = 0.5
    overstretched_threshold = nominal_link_length * args.overstretched_factor_threshold
    for i in range(args.n):
        # half gaussian with variance such that ~50% of ropes will be overstretched
        length = abs(np.random.randn()) * 0.042 + nominal_link_length
        rope_configurations[i] = link_bot_pycommon.make_random_rope_configuration(sdf_data.extent, length=length)
        tail_x = rope_configurations[i, 0]
        tail_y = rope_configurations[i, 1]
        mid_x = rope_configurations[i, 2]
        mid_y = rope_configurations[i, 3]
        head_x = rope_configurations[i, 4]
        head_y = rope_configurations[i, 5]
        row, col = link_bot_pycommon.point_to_sdf_idx(head_x, head_y, sdf_resolution, sdf_origin)
        constraint_labels[i, 0] = sdf[row, col] < args.sdf_threshold
        tail_mid_overstretched = np.hypot(tail_x - mid_x, tail_y - mid_y) > overstretched_threshold
        mid_head_overstretched = np.hypot(mid_x - head_x, mid_y - head_y) > overstretched_threshold
        constraint_labels[i, 1] = tail_mid_overstretched or mid_head_overstretched

    n_positive = np.count_nonzero(np.any(constraint_labels, axis=1))
    percentage_positive = n_positive * 100.0 / constraint_labels.shape[0]

    if args.n_plots and args.n_plots > 0:
        for i in np.random.choice(rope_configurations.shape[0], size=args.n_plots):
            plot(args, sdf_data, args.sdf_threshold, rope_configurations[i], constraint_labels[i])

        plt.show()

    return rope_configurations, constraint_labels, sdf_data, percentage_positive


def generate(args):
    if args.outdir:
        if os.path.isfile(args.outdir):
            print(Fore.RED + "argument outdir is an existing file, aborting." + Fore.RESET)
            return
        elif not os.path.isdir(args.outdir):
            os.mkdir(args.outdir)

    if not args.seed:
        # I know this looks crazy, but the idea is that when we run the script multiple times we don't want to get the same output
        # but we als do want to be able to recreate the output from a seed, so we generate a random seed if non is provided
        args.seed = np.random.randint(0, 10000)
    np.random.seed(args.seed)

    # Define what kinds of labels are contained in this dataset
    constraint_label_types = [LabelType.SDF, LabelType.Overstretching]

    filename_pairs = []
    percentages_positive = []
    for i in range(args.m):
        rope_configurations, constraint_labels, sdf_data, percentage_violation = generate_env(args)
        percentages_positive.append(percentage_violation)
        if args.outdir:
            rope_data_filename = os.path.join(args.outdir, 'rope_data_{:d}.npz'.format(i))
            sdf_filename = os.path.join(args.outdir, 'sdf_data_{:d}.npz'.format(i))

            # FIXME: order matters
            filename_pairs.append([sdf_filename, rope_data_filename])

            np.savez(rope_data_filename,
                     rope_configurations=rope_configurations,
                     constraints=constraint_labels,
                     # save this so we can visualize what the true constraint boundary is
                     # this value should not be used for training, that would be cheating!
                     threshold=args.sdf_threshold)
            sdf_data.save(sdf_filename)
        print(".", end='')
        sys.stdout.flush()
    print("done")

    mean_percentage_positive = np.mean(percentages_positive)
    print("Class balance: mean % positive: {}".format(mean_percentage_positive))

    if args.outdir:
        dataset_filename = os.path.join(args.outdir, 'dataset.json')
        dataset = MultiEnvironmentDataset(filename_pairs, constraint_label_types=constraint_label_types,
                                          n_obstacles=args.n_obstacles, obstacle_size=args.obstacle_size,
                                          threshold=args.sdf_threshold, seed=args.seed)
        dataset.save(dataset_filename)


def plot_main(args):
    np.random.seed(args.seed)
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    generator = dataset.generator(args.n)
    xs, ys = generator[0]
    for i in range(args.n):
        sdf_data = SDF(sdf=np.squeeze(xs['sdf'][i]),
                       gradient=xs['sdf_gradient'][i],
                       resolution=xs['sdf_resolution'][i],
                       origin=xs['sdf_origin'][i])
        rope_configuration = xs['rope_configuration'][i]
        constraint_labels = ys['all_output'][i]
        plot(args, sdf_data, dataset.threshold, rope_configuration, constraint_labels)

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    generate_parser = subparsers.add_parser('generate')
    generate_parser.set_defaults(func=generate)
    generate_parser.add_argument('n', type=int, help='number of data points per environment')
    generate_parser.add_argument('m', type=int, help='number of environments')
    generate_parser.add_argument('w', type=int, help='environment with in meters (int)')
    generate_parser.add_argument('h', type=int, help='environment with in meters (int)')
    generate_parser.add_argument('--seed', type=int, help='random seed')
    generate_parser.add_argument('--res', '-r', type=float, default=0.05, help='size of cells in meters')
    generate_parser.add_argument('--n-obstacles', type=int, default=14, help='size of obstacles in cells')
    generate_parser.add_argument('--obstacle-size', type=int, default=8, help='size of obstacles in cells')
    generate_parser.add_argument('--sdf-threshold', type=np.float32, default=0.0)
    generate_parser.add_argument('--overstretched-factor-threshold', type=np.float32, default=1.1)
    generate_parser.add_argument('--n-plots', type=int, help='number of examples to plot')
    generate_parser.add_argument('--outdir')

    plot_parser = subparsers.add_parser('plot')
    plot_parser.set_defaults(func=plot_main)
    plot_parser.add_argument('dataset', help='json dataset file')
    plot_parser.add_argument('n', type=int, help='number of examples to plot')
    plot_parser.add_argument('--seed', type=int, help='random seed')

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
