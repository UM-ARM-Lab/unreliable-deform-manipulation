#!/usr/bin/env python

import argparse
import json
import pathlib
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from colorama import Style
from tabulate import tabulate

from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.metric_utils import make_row


def invert_dict(data: List) -> Dict:
    d = {}
    for di in data:
        for k, v in di.items():
            if k not in d:
                d[k] = []
            d[k].append(v)
    return d


def error_metrics(planning_times, path_length, final_errors):
    return np.array([make_row('planning time (s)', planning_times),
                     make_row('path length (m)', path_length),
                     make_row('final error (m)', final_errors),
                     ], dtype=np.object)


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_dir', help='foldering containing folders containing metrics.jsons files', type=pathlib.Path)

    args = parser.parse_args()

    subfolders = args.results_dir.iterdir()
    for subfolder in subfolders:
        metrics_filename = subfolder / 'metrics.json'
        metrics = json.load(metrics_filename.open("r"))
        data = metrics.pop('metrics')
        meta = metrics

        data = invert_dict(data)
        planning_times = data['planning_time']
        path_length = data['path_length']
        final_errors = data['final_error']

        headers = ['error metric', 'min', 'max', 'mean', 'median', 'std']
        aggregate_metrics = error_metrics(planning_times, path_length, final_errors)
        table = tabulate(aggregate_metrics, headers=headers, tablefmt='github', floatfmt='6.4f')
        print(Style.BRIGHT + "{}:".format(subfolder.name) + Style.RESET_ALL)
        print(table)

        plt.title(subfolder.name)
        plt.hist(final_errors)


if __name__ == '__main__':
    main()
