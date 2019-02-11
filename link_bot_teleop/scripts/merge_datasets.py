#!/usr/bin/env python
from __future__ import print_function

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs='*')
    parser.add_argument("outfile", help='merged output dataset')

    args = parser.parse_args()

    datasets = []
    for datasetname in args.datasets:
        dataset = np.load(datasetname)
        print(dataset.shape)
        datasets.append(dataset)

    merged_data = np.concatenate(datasets, axis=0)

    np.save(args.outfile, merged_data)


if __name__ == '__main__':
    main()
