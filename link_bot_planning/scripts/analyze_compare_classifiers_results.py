#!/usr/bin/env python

import argparse
import json
import pathlib
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from colorama import Style
from scipy import stats
from tabulate import tabulate

from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.metric_utils import make_row


def dict_to_pvale_table(data_dict):
    pvalues = np.zeros([len(data_dict), len(data_dict)])
    for i, (name1, e1) in enumerate(data_dict.items()):
        for j, (name2, e2) in enumerate(data_dict.items()):
            _, pvalue = stats.ttest_ind(e1, e2)
            pvalues[i, j] = pvalue
    headers = [''] + list(data_dict.keys())
    pvalues
    table = tabulate(pvalues, headers=headers, tablefmt='github', floatfmt='6.4f')
    # table[1, 0] = ''
    return table


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
    final_errors_comparisons = {}
    for subfolder in subfolders:
        if not subfolder.is_dir():
            continue
        metrics_filename = subfolder / 'metrics.json'
        metrics = json.load(metrics_filename.open("r"))
        data = metrics.pop('metrics')

        data = invert_dict(data)
        planning_times = data['planning_time']
        path_length = data['path_length']
        final_errors = data['final_execution_error']
        final_errors_comparisons[str(subfolder.name)] = final_errors

        headers = ['error metric', 'min', 'max', 'mean', 'median', 'std']
        aggregate_metrics = error_metrics(planning_times, path_length, final_errors)
        table = tabulate(aggregate_metrics, headers=headers, tablefmt='github', floatfmt='6.4f')
        print(Style.BRIGHT + "{}:".format(subfolder.name) + Style.RESET_ALL)
        print(table)

        plt.title(subfolder.name)
        plt.hist(final_errors)

    print(Style.BRIGHT + "p-value matrix" + Style.RESET_ALL)
    print(dict_to_pvale_table(final_errors_comparisons))


if __name__ == '__main__':
    main()
