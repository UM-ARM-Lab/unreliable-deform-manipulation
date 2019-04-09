#!/usr/bin/env python
from __future__ import print_function

# This script did NOT generate fake_linear1-6.txt
import argparse
import matplotlib.pyplot as plt
import numpy as np

true_fake_B = np.array([[0.1, 0.2], [0.3, 0.4]])
true_fake_C = np.array([[2, 1], [0, 3]])

def generate(args):
    data = np.ndarray((args.num_trajectories, args.trajectory_length, 7))

    for t in range(args.num_trajectories):
        # uniformly randomly sample a starting point, velocity, and direction
        v = np.random.uniform(0, args.maxv)
        angle = np.random.uniform(-np.pi, np.pi)
        o = np.random.uniform(-5, 5, size=2)
        vx = np.cos(angle) * v
        vy = np.sin(angle) * v
        u = np.array([vx, vy])
        for s in range(args.trajectory_length):
            # forward integrate dynamics
            o = o + args.dt * np.dot(true_fake_B, o) + args.dt * np.dot(true_fake_C, u)
            time = s * args.dt
            data[t][s] = [time, o[0], o[1], o[0] + o[1], 2 * o[0] - 4 * o[1], vx, vy]

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
