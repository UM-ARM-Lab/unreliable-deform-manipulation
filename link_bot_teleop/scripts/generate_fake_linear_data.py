#!/usr/bin/env python
from __future__ import print_function

# This script did NOT generate fake_linear1-6.txt
import argparse
import matplotlib.pyplot as plt
import numpy as np


def generate(args):
    data = np.ndarray((args.num_trajectories, args.trajectory_length, 5))

    for t in range(args.num_trajectories):
        # uniformly randomly sample a starting point, velocity, and direction
        v = np.random.uniform(0, args.maxv)
        angle = np.random.uniform(-np.pi, np.pi)
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        vx = np.cos(angle) * v
        vy = np.sin(angle) * v
        for s in range(args.trajectory_length):
            # forward integrate dynamics
            x = x + vx * args.dt
            y = y + vy * args.dt
            time = s * args.dt
            data[t][s] = [time, x, y, x + y, 2 * x - 4 * y]

    np.save(args.outfile, data)


def plot(args):
    data = np.load(args.infile)
    for t in range(args.max_plots):
        trajectory = data[t]
        plt.figure()
        for i in range(1, trajectory.shape[1]):
            plt.plot(trajectory[:, 0], trajectory[:, i], label="s[{}]".format(i))
        plt.title("Trajectory {}".format(t))
        plt.xlabel("time (s)")
        plt.ylabel("state value")
        plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument("num_trajectories", type=int, help='number of trajectories')
    generate_parser.add_argument("trajectory_length", type=int, help='length of trajectories')
    generate_parser.add_argument("outfile", help='output file name')
    generate_parser.add_argument("--dt", type=float, help='dt in seconds', default=0.1)
    generate_parser.add_argument("--maxv", type=float, help='maximum velocity in m/s', default=1.0)
    generate_parser.set_defaults(func=generate)
    plot_parser = subparsers.add_parser('plot')
    plot_parser.add_argument("infile", help='a previously generate data file to plot')
    plot_parser.add_argument("--max-plots", type=int, help="only plot this many trajectories", default=10)
    plot_parser.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
