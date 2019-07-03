#!/usr/bin/env python
from __future__ import print_function

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs='*')
    parser.add_argument("outfile", help='merged output dataset')

    args = parser.parse_args()

    if len(args.datasets) == 1:
        print("ERROR: You must supply two datasets! The last argument is interpreted as the output filename.")
        return

    data_dict = {}
    for datasetname in args.datasets:
        dataset = np.load(datasetname)
        for key, data in dataset.items():
            if key not in data_dict:
                data_dict[key] = data
            else:
                data_dict[key] = np.concatenate([data_dict[key], data], axis=0)

    np.savez(args.outfile, **data_dict)


if __name__ == '__main__':
    main()
