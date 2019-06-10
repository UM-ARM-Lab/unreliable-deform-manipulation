#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import argparse
from link_bot_pycommon import link_bot_pycommon
import sdf_tools


def plot(grid_world, sdf, sdf_gradient, rope_configurations):
    sdf_fig = plt.figure()
    plt.imshow(sdf)

    grad_fig = plt.figure()
    plt.imshow(sdf)
    n_rows, n_cols = sdf.shape
    subsample = 16
    x, y = np.meshgrid(range(0, n_rows, subsample), range(0, n_cols, subsample))
    dx = sdf_gradient[::subsample, ::subsample, 0]
    dy = sdf_gradient[::subsample, ::subsample, 1]
    plt.quiver(x, y, dx, dy, units='x', scale=0.1, headwidth=1, headlength=3)

    grid_fig = plt.figure()
    plt.imshow(grid_world)

    for rope_configuration in rope_configurations:
        xs = [rope_configuration[0], rope_configuration[2], rope_configuration[4]]
        ys = [rope_configuration[1], rope_configuration[3], rope_configuration[5]]
        plt.plot(xs, ys, linewidth=1, zorder=1)
        plt.scatter(rope_configuration[4], rope_configuration[5], s=4, zorder=2)

    return (grid_fig, sdf_fig, grad_fig)


def generate(args):
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
    sdf_origin = [0, 0]
    sdf, sdf_gradient = sdf_tools.compute_2d_sdf_and_gradient(grid_world, args.res, sdf_origin)

    # create random rope configurations
    rope_configurations = np.ndarray((args.n, 6))
    for i in range(args.n):
        theta_1 = np.random.uniform(-np.pi, np.pi)
        theta_2 = np.random.uniform(-np.pi, np.pi)
        head_x = np.random.uniform(2, args.w - 2)
        head_y = np.random.uniform(2, args.h - 2)
        rope_configurations[i] = link_bot_pycommon.make_rope_configuration(head_x, head_y, theta_1, theta_2)

    if args.outfile:
        res_arr = np.array([args.res, args.res])
        np.savez(args.outfile,
                 grid_world=grid_world,
                 sdf=sdf,
                 sdf_gradient=sdf_gradient,
                 sdf_origin=sdf_origin,
                 sdf_resolution=res_arr)

    if args.plot:
        plot(grid_world, sdf, sdf_gradient, rope_configurations)

        plt.show()


def plot_main(args):
    data = np.load(args.data)
    grid_world = data['grid_world']
    rope_configurations = data['rope_configurations']

    plot(grid_world, rope_configurations)

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    generate_parser = subparsers.add_parser('generate')
    generate_parser.set_defaults(func=generate)
    generate_parser.add_argument('n', type=int, default=1000, help='number of data points')
    generate_parser.add_argument('w', type=int, default=10, help='environment with in meters (int)')
    generate_parser.add_argument('h', type=int, default=10, help='environment with in meters (int)')
    generate_parser.add_argument('--res', '-r', type=float, default=0.01, help='size of cells in meters')
    generate_parser.add_argument('--n-obstacles', type=int, default=10, help='size of obstacles in cells')
    generate_parser.add_argument('--obstacle-size', type=int, default=10, help='size of obstacles in cells')
    generate_parser.add_argument('--plot', action='store_true')
    generate_parser.add_argument('--outfile')

    plot_parser = subparsers.add_parser('plot')
    plot_parser.set_defaults(func=plot_main)
    plot_parser.add_argument('data', help='generated file, npz')

    args = parser.parse_args()
    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
