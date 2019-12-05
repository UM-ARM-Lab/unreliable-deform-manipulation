#!/usr/bin/env python

import argparse
import plotly.graph_objects as go
import json
import pathlib
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from colorama import Style
from scipy import stats
from tabulate import tabulate

from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.metric_utils import row_stats


def dict_to_pvale_table(data_dict: Dict, table_format: str):
    pvalues = np.zeros([len(data_dict), len(data_dict) + 1], dtype=object)
    for i, (name1, e1) in enumerate(data_dict.items()):
        pvalues[i, 0] = name1
        for j, (_, e2) in enumerate(data_dict.items()):
            _, pvalue = stats.ttest_ind(e1, e2)
            pvalues[i, j + 1] = pvalue
    headers = [''] + list(data_dict.keys())
    table = tabulate(pvalues, headers=headers, tablefmt=table_format, floatfmt='6.4f')
    return table


def invert_dict(data: List) -> Dict:
    d = {}
    for di in data:
        for k, v in di.items():
            if k not in d:
                d[k] = []
            d[k].append(v)
    return d


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_dir', help='foldering containing folders containing metrics.jsons files', type=pathlib.Path)
    parser.add_argument('--table-format', choices=['latex', 'github'], default='github', type=str, help='printing format')

    args = parser.parse_args()

    subfolders = args.results_dir.iterdir()
    final_errors_comparisons = {}
    aggregate_metrics = {
        'planning time': [['min', 'max', 'mean', 'median', 'std']],
        'final tail error': [['min', 'max', 'mean', 'median', 'std']],
    }

    headers = ['']
    for subfolder in subfolders:
        if not subfolder.is_dir():
            continue
        metrics_filename = subfolder / 'metrics.json'
        metrics = json.load(metrics_filename.open("r"))
        data = metrics.pop('metrics')

        data = invert_dict(data)
        planning_times = data['planning_time']
        final_errors = data['final_execution_error']
        final_errors_comparisons[str(subfolder.name)] = final_errors
        headers.append(str(subfolder.name))

        aggregate_metrics['planning time'].append(row_stats(planning_times))
        aggregate_metrics['final tail error'].append(row_stats(final_errors))

        plt.title(subfolder.name)
        plt.hist(final_errors)

    for metric_name, table_data in aggregate_metrics.items():
        data = [go.Table(name=metric_name,
                         header={'values': headers,
                                 'font_size': 18},
                         cells={'values': table_data,
                                'format': [None, '5.3f', '5.3f', '5.3f'],
                                'font_size': 14})]
        fig = go.Figure(data)
        outfile = args.results_dir / '{}_table.png'.format(metric_name)
        fig.write_image(str(outfile), scale=4)

    print(Style.BRIGHT + "p-value matrix" + Style.RESET_ALL)
    print(dict_to_pvale_table(final_errors_comparisons, table_format=args.table_format))


if __name__ == '__main__':
    main()
