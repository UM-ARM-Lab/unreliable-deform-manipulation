#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np

from src.link_bot.link_bot_sdf_tools.src.link_bot_sdf_tools.link_bot_sdf_tools import SDF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sdf_files', nargs='+')
    parser.add_argument('--threshold', type=float, default=0.0)

    args = parser.parse_args()

    for sdf_file in args.sdf_files:
        sdf_data = SDF.load(sdf_file)

        binary_image = sdf_data.sdf > args.threshold
        plt.figure()
        plt.title(sdf_file)
        plt.imshow(np.flipud(binary_image.T), extent=sdf_data.extent)

    plt.show()


if __name__ == '__main__':
    main()

