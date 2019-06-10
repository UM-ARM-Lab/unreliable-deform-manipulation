#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import argparse
from link_bot_pycommon import link_bot_pycommon
import sdf_tools


def plot(args, grid_world, sdf, sdf_gradient, rope_configurations, constraint_labels):
    sdf_fig = plt.figure()
    # Note: images should always be flipped and transposed because of how image coordinates work.
    #       however, we do not flip/transpose when indexing into the SDF, this is just needed when plotting.
    plt.imshow(np.flipud(sdf.T), extent=[0, args.w, 0, args.h])

    grad_fig = plt.figure()
    plt.imshow(np.flipud(sdf.T), extent=[0, args.w, 0, args.h])
    n_rows, n_cols = sdf.shape
    subsample = 16
    x, y = np.meshgrid(range(0, n_rows, subsample), range(0, n_cols, subsample))
    dx = sdf_gradient[::subsample, ::subsample, 0]
    dy = sdf_gradient[::subsample, ::subsample, 1]
    plt.quiver(x * args.res, y * args.res, dx, dy, units='x', scale=5, headwidth=2, headlength=4)

    grid_fig = plt.figure()
    plt.imshow(np.flipud(grid_world.T), extent=[0, args.w, 0, args.h])

    for rope_configuration, constraint_label in zip(rope_configurations, constraint_labels):
        xs = [rope_configuration[0], rope_configuration[2], rope_configuration[4]]
        ys = [rope_configuration[1], rope_configuration[3], rope_configuration[5]]
        plt.plot(xs, ys, linewidth=1, zorder=2)
        plt.scatter(rope_configuration[4], rope_configuration[5], s=4, zorder=3)
        color = 'r' if constraint_label else 'g'
        plt.scatter(rope_configuration[4], rope_configuration[5], s=32, c=color, zorder=1)

    return grid_fig, sdf_fig, grad_fig


def generate(args):
    np.random.seed(args.seed)

    sdf_origin = np.array([0.0, 0.0], dtype=np.int32)
    res_arr = np.array([args.res, args.res], dtype=np.float32)

    n_rows = int(args.h / args.res)
    n_cols = int(args.w / args.res)
    n_cells = n_rows * n_cols

    grid_world = np.zeros((n_rows, n_cols))
    occupied_cells = np.random.choice(n_cells, size=args.n_obstacles)
    occupied_cells_row, occupied_cells_col = np.unravel_index(occupied_cells, [n_rows, n_cols])
    for obstacle_row, obstacle_col in zip(occupied_cells_row, occupied_cells_col):
        for drow in range(-args.obstacle_size, args.obstacle_size + 1):
            for dcol in range(-args.obstacle_size, args.obstacle_size + 1):
                r = (obstacle_row + int(drow)) % n_rows
                c = (obstacle_col + int(dcol)) % n_cols
                grid_world[r, c] = 1

    # create a signed distance field from the grid
    sdf, sdf_gradient = sdf_tools.compute_2d_sdf_and_gradient(grid_world, args.res, sdf_origin)

    # create random rope configurations
    rope_configurations = np.ndarray((args.n, 6), dtype=np.float32)
    constraint_labels = np.ndarray((args.n, 1), dtype=np.float32)
    for i in range(args.n):
        theta_1 = np.random.uniform(-np.pi, np.pi)
        theta_2 = np.random.uniform(-np.pi, np.pi)
        head_x = np.random.uniform(2, args.w - 2)
        head_y = np.random.uniform(2, args.h - 2)
        rope_configurations[i] = link_bot_pycommon.make_rope_configuration(head_x, head_y, theta_1, theta_2)
        row, col = link_bot_pycommon.point_to_sdf_idx(head_x, head_y, res_arr, sdf_origin)
        constraint_labels[i] = sdf[row, col] < args.distance_constraint_threshold

    n_positive = np.count_nonzero(constraint_labels)
    percentage_positive = n_positive * 100.0 / args.n
    print("% positive examples: {}".format(percentage_positive))

    if args.outfile:
        np.savez(args.outfile + '_data.npz',
                 grid_world=grid_world,
                 states=rope_configurations,
                 rope_configurations=rope_configurations,
                 constraints=constraint_labels)
        np.savez(args.outfile + '_sdf.npz',
                 sdf=sdf,
                 sdf_gradient=sdf_gradient,
                 sdf_origin=sdf_origin,
                 sdf_resolution=res_arr)

    if args.plot:
        plot(args, grid_world, sdf, sdf_gradient, rope_configurations, constraint_labels)

        plt.show()


def plot_main(args):
    data = np.load(args.data)
    grid_world = data['grid_world']
    rope_configurations = data['rope_configurations']
    constraint_labels = data['constraints']

    sdf, sdf_gradient, sdf_resolution, sf_origin = link_bot_pycommon.load_sdf(args.sdf)

    args.h = sdf_resolution[0] * sdf.shape[0]
    args.w = sdf_resolution[1] * sdf.shape[1]
    args.res = sdf_resolution[0]
    plot(args, grid_world, sdf, sdf_gradient, rope_configurations, constraint_labels)

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    generate_parser = subparsers.add_parser('generate')
    generate_parser.set_defaults(func=generate)
    generate_parser.add_argument('n', type=int, help='number of data points')
    generate_parser.add_argument('w', type=int, help='environment with in meters (int)')
    generate_parser.add_argument('h', type=int, help='environment with in meters (int)')
    generate_parser.add_argument('--seed', type=int, default=2, help='random seed')
    generate_parser.add_argument('--res', '-r', type=float, default=0.01, help='size of cells in meters')
    generate_parser.add_argument('--n-obstacles', type=int, default=69, help='size of obstacles in cells')
    generate_parser.add_argument('--obstacle-size', type=int, default=50, help='size of obstacles in cells')
    generate_parser.add_argument('--distance-constraint-threshold', type=np.float32, default=0.0, help='constraint threshold')
    generate_parser.add_argument('--plot', action='store_true')
    generate_parser.add_argument('--outfile')

    plot_parser = subparsers.add_parser('plot')
    plot_parser.set_defaults(func=plot_main)
    plot_parser.add_argument('data', help='generated file, npz')
    plot_parser.add_argument('sdf', help='generated file, npz')

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
