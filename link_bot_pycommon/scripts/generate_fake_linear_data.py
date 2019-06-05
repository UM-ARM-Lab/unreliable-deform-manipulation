#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile")
    parser.add_argument("-N", type=int, default=1000)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    data = np.random.randn(args.N, 12)

    tail_x = 5
    tail_y = 5
    for i in range(args.N):
        # take a random action
        head_vx = tail_x + np.random.randn() * 0.5
        head_vy = tail_y + np.random.randn() * 0.5

        data[i][0] = tail_x
        data[i][1] = tail_y
        data[i][10] = head_vx
        data[i][11] = head_vy

        tail_x += 0.1 * head_vx + np.random.randn() * 1e-3
        tail_y += 0.1 * head_vy + np.random.randn() * 1e-3

    if args.plot:
        plt.plot(data[:, 10], label='head_vx')
        plt.plot(data[:, 11], label='head_vy')
        plt.title("head velocity")
        plt.xlabel("time (steps")
        plt.ylabel("y (m)")
        plt.legend()

        plt.figure()
        plt.plot(data[:, 0], data[:, 1])
        plt.scatter(data[:, 0], data[:, 1])
        plt.title("tail position")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.show()

    np.savetxt(args.outfile, data)


if __name__ == '__main__':
    main()
