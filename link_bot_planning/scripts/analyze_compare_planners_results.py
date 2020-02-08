#!/usr/bin/env python

import argparse
import json
import pathlib
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from colorama import Style
from scipy import stats
from tabulate import tabulate

from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.link_bot_pycommon import transpose_2d_lists
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
        'execution_to_goal_errors': [['min', 'max', 'mean', 'median', 'std']],
        'plan_to_goal_errors': [['min', 'max', 'mean', 'median', 'std']],
        'execution_to_plan_errors': [['min', 'max', 'mean', 'median', 'std']],
        'num_nodes': [['min', 'max', 'mean', 'median', 'std']],
    }

    execution_to_goal_errors_comparisons = {}
    execution_to_plan_errors_comparisons = {}
    errors_thresholds = np.linspace(0.1, 1.0, 10)
    print('-' * 90)
    if not args.no_plot:
        plt.figure()
        execution_ax = plt.gca()
        execution_ax.set_xlabel("Success Threshold, Final Tail Error")
        execution_ax.set_ylabel("Success Rate")
        execution_ax.set_ylim([0, 100])
        execution_ax.set_title("Success In Execution")

        plt.figure()
        planning_ax = plt.gca()
        planning_ax.set_xlabel("Success Threshold, Final Tail Error")
        planning_ax.set_ylabel("Success Rate")
        planning_ax.set_ylim([0, 100])
        planning_ax.set_title("Success In Planning")

    for results_dir in args.results_dirs:
        subfolders = results_dir.iterdir()

        for subfolder in subfolders:
            if not subfolder.is_dir():
                continue
            metrics_filename = subfolder / 'metrics.json'
            metrics = json.load(metrics_filename.open("r"))
            timeout = metrics['planner_params']['timeout']
            data = metrics.pop('metrics')

            data = invert_dict(data)
            planning_times = np.array(data['planning_time'])
            # TODO: rename these keys
            execution_to_goal_errors = data['final_execution_error']
            plan_to_goal_errors = data['final_planning_error']
            has_execution_to_plan_error = ('execution_to_plan_error' in data)
            if has_execution_to_plan_error:
                execution_to_plan_errors = data['execution_to_plan_error']
            num_nodes = data['num_nodes']
            timeouts = np.sum((planning_times > timeout).astype(np.int))
            timeout_percentage = timeouts / planning_times.shape[0] * 100
            name = subfolder.name
            if not args.no_plot:
                execution_successes = []
                for threshold in errors_thresholds:
                    execution_successes.append(np.count_nonzero(execution_to_goal_errors < threshold))
                execution_ax.plot(errors_thresholds, execution_successes, label=name)

                if has_execution_to_plan_error:
                    planning_successes = []
                    for threshold in errors_thresholds:
                        planning_successes.append(np.count_nonzero(execution_to_plan_errors < threshold))
                    planning_ax.plot(errors_thresholds, planning_successes, label=name)

            execution_to_goal_errors_comparisons[str(subfolder.name)] = execution_to_goal_errors
            execution_to_plan_errors_comparisons[str(subfolder.name)] = plan_to_goal_errors
            headers.append(str(subfolder.name))

            aggregate_metrics['planning_time'].append(row_stats(planning_times))
            aggregate_metrics['execution_to_goal_errors'].append(row_stats(execution_to_goal_errors))
            if has_execution_to_plan_error:
                aggregate_metrics['execution_to_plan_errors'].append(row_stats(execution_to_plan_errors))
            aggregate_metrics['plan_to_goal_errors'].append(row_stats(plan_to_goal_errors))
            aggregate_metrics['num_nodes'].append(row_stats(num_nodes))

            print("{:50s}: {:3.2f}% timeout ".format(str(subfolder), timeout_percentage))

    plt.legend()

    print('-' * 90)

    for metric_name, table_data in aggregate_metrics.items():

        print(Style.BRIGHT + metric_name + Style.RESET_ALL)
        table_data_flipped = transpose_2d_lists(table_data)
        table = tabulate(table_data_flipped, headers=headers, tablefmt='github', floatfmt='6.4f')
        print(table)
        print()

        if not args.no_plot:
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
            fig.show()

    print(Style.BRIGHT + "p-value matrix (vs execution)" + Style.RESET_ALL)
    print(dict_to_pvale_table(execution_to_goal_errors_comparisons, table_format='github'))
    print(Style.BRIGHT + "p-value matrix (vs plan)" + Style.RESET_ALL)
    print(dict_to_pvale_table(execution_to_plan_errors_comparisons, table_format='github'))

    plt.savefig('results/final_tail_error_hist.png')
    if not args.no_plot:
        plt.show()


if __name__ == '__main__':
    main()