#!/usr/bin/env python

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=pathlib.Path, help='directory containing a folder "failures"')

    args = parser.parse_args()

    failures = args.comparison / 'failures'
    for trial in failures.iterdir():
        info_filename = trial / "info.json"
        image_filename = trial / "full_sdf.png"

        i = Image.open(image_filename)

        d = json.load(open(info_filename, 'r'))
        p = d['start'][0]

        plt.imshow(i, extent=d['sdf']['extent'])
        plt.scatter([p[0], p[2], p[4]], [p[1], p[3], p[5]])
        plt.show()


if __name__ == '__main__':
    main()
