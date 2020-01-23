#!/usr/bin/env python

import argparse

import plotly
import plotly.graph_objects as go
import json
import pathlib
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from colorama import Style
from scipy import stats
from tabulate import tabulate

from link_bot_planning.params import PlannerParams
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
    parser.add_argument('results_dirs', help='folders containing folders containing metrics.json', type=pathlib.Path, nargs='+')
    parser.add_argument('--no-plot', action='store_true')

    args = parser.parse_args()

    headers = ['']
    aggregate_metrics = {
        'planning_time': [['min', 'max', 'mean', 'median', 'std']],
        'final_tail_error': [['min', 'max', 'mean', 'median', 'std']],
        'num_nodes': [['min', 'max', 'mean', 'median', 'std']],
        'timeouts': [['min', 'max', 'mean', 'median', 'std']],
    }
    plt.figure()
    final_errors_comparisons = {}
    for results_dir in args.results_dirs:
        subfolders = results_dir.iterdir()

        for subfolder in subfolders:
            if not subfolder.is_dir():
                continue
            metrics_filename = subfolder / 'metrics.json'
            metrics = json.load(metrics_filename.open("r"))
            timeout = json.loads(metrics['planner_params'])['timeout']
            data = metrics.pop('metrics')

            data = invert_dict(data)
            planning_times = np.array(data['planning_time'])
            final_errors = data['final_execution_error']
            num_nodes = data['num_nodes']
            timeouts = (planning_times > timeout).astype(np.int)
            name = subfolder.name
            n, x, _ = plt.hist(final_errors, bins=np.linspace(0, 3, 15), alpha=0)
            bin_centers = 0.5 * (x[1:] + x[:-1])
            plt.plot(bin_centers, n, label=name)
            final_errors_comparisons[str(subfolder.name)] = final_errors
            headers.append(str(subfolder.name))

            aggregate_metrics['planning_time'].append(row_stats(planning_times))
            aggregate_metrics['final_tail_error'].append(row_stats(final_errors))
            aggregate_metrics['num_nodes'].append(row_stats(num_nodes))
            aggregate_metrics['timeouts'].append(row_stats(timeouts))

    plt.legend()

    for metric_name, table_data in aggregate_metrics.items():
        data = [go.Table(name=metric_name,
                         header={'values': headers,
                                 'font_size': 18},
                         cells={'values': table_data,
                                'format': [None, '5.3f', '5.3f', '5.3f'],
                                'font_size': 14})]
        layout = go.Layout(title=metric_name)
        fig = go.Figure(data, layout)
        outfile = pathlib.Path('results') / '{}_table.png'.format(metric_name)
        fig.write_image(str(outfile), scale=4)
        if not args.no_plot:
            fig.show()

    print(Style.BRIGHT + "p-value matrix" + Style.RESET_ALL)
    print(dict_to_pvale_table(final_errors_comparisons, table_format='github'))

    plt.savefig('results/final_tail_error_hist.png')
    if not args.no_plot:
        plt.show()


if __name__ == '__main__':
    main()
