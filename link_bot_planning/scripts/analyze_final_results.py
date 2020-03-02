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
from link_bot_pycommon.link_bot_pycommon import transpose_2d_lists
from link_bot_pycommon.metric_utils import row_stats


def dict_to_pvale_table(data_dict: Dict, table_format: str, fmt: str = '{:5.3f}'):
    pvalues = np.zeros([len(data_dict), len(data_dict) + 1], dtype=object)
    for i, (name1, e1) in enumerate(data_dict.items()):
        pvalues[i, 0] = name1
        for j, (_, e2) in enumerate(data_dict.items()):
            _, pvalue = stats.ttest_ind(e1, e2)
            if pvalue < 0.01:
                prefix = "! "
            else:
                prefix = "  "
            pvalues[i, j + 1] = prefix + fmt.format(pvalue)
    headers = [''] + list(data_dict.keys())
    table = tabulate(pvalues, headers=headers, tablefmt=table_format)
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
    plt.style.use('paper')

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_dirs', help='folders containing folders containing metrics.json', type=pathlib.Path, nargs='+')
    parser.add_argument('--no-plot', action='store_true')

    args = parser.parse_args()

    headers = ['']
    aggregate_metrics = {
        'planning_time': [['min', 'max', 'mean', 'median', 'std']],
        'execution_to_goal_errors': [['min', 'max', 'mean', 'median', 'std']],
        'plan_to_goal_errors': [['min', 'max', 'mean', 'median', 'std']],
        'mean_plan_to_execution_errors': [['min', 'max', 'mean', 'median', 'std']],
        'final_plan_to_execution_errors': [['min', 'max', 'mean', 'median', 'std']],
        'num_nodes': [['min', 'max', 'mean', 'median', 'std']],
        'path_length': [['min', 'max', 'mean', 'median', 'std']],
    }

    execution_to_goal_errors_comparisons = {}
    plan_to_execution_errors_comparisons = {}
    max_error = 1.5
    errors_thresholds = np.linspace(0.05, max_error, 49)
    print('-' * 90)
    if not args.no_plot:
        plt.figure()
        execution_ax = plt.gca()
        execution_ax.set_xlabel("Success Threshold, Final Tail Error")
        execution_ax.set_ylabel("Success Rate")
        execution_ax.set_ylim([-0.1, 100.1])
        execution_ax.set_title("Success In Execution")

        plt.figure()
        planning_ax = plt.gca()
        planning_ax.set_xlabel("Success Threshold, Final Tail Error")
        planning_ax.set_ylabel("Success Rate")
        planning_ax.set_ylim([-0.1, 100.1])
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
            N = len(data)
            print("{} has {} examples".format(subfolder, N))

            data = invert_dict(data)
            planning_times = np.array(data['planning_time'])
            path_length = np.array(data['planning_time'])
            mean_plan_to_execution_errors = []
            for planned, actual in zip(data['planned_path'], data['actual_path']):
                planned_path = np.array(planned)
                actual_path = np.array(actual)
                # FIXME: old results included an extra state, we should rerun experiments
                error = np.linalg.norm(planned_path[:-1, 0:2] - actual_path[:-1, 0:2], axis=1)
                mean_plan_to_execution_errors.append(np.mean(error))
            # TODO: rename these keys
            execution_to_goal_errors = data['final_execution_error']
            plan_to_goal_errors = data['final_planning_error']
            has_plan_to_execution_error = ('plan_to_execution_error' in data)
            if has_plan_to_execution_error:
                final_plan_to_execution_errors = data['plan_to_execution_error']
            num_nodes = data['num_nodes']
            timeouts = np.sum((planning_times > timeout).astype(np.int))
            timeout_percentage = timeouts / planning_times.shape[0] * 100
            name = str(subfolder.name).replace('_', ' ')
            if not args.no_plot:
                execution_successes = []
                for threshold in errors_thresholds:
                    success_percentage = np.count_nonzero(execution_to_goal_errors < threshold) / N * 100
                    execution_successes.append(success_percentage)
                execution_ax.plot(errors_thresholds, execution_successes, label=name, linewidth=5)

                if has_plan_to_execution_error:
                    planning_successes = []
                    for threshold in errors_thresholds:
                        success_percentage = np.count_nonzero(final_plan_to_execution_errors < threshold) / N * 100
                        planning_successes.append(success_percentage)
                    planning_ax.plot(errors_thresholds, planning_successes, label=name, linewidth=5)

            execution_to_goal_errors_comparisons[str(subfolder.name)] = execution_to_goal_errors
            if has_plan_to_execution_error:
                plan_to_execution_errors_comparisons[str(subfolder.name)] = final_plan_to_execution_errors
            headers.append(str(subfolder.name))

            aggregate_metrics['planning_time'].append(row_stats(planning_times))
            aggregate_metrics['path_length'].append(row_stats(path_length))
            aggregate_metrics['mean_plan_to_execution_errors'].append(row_stats(mean_plan_to_execution_errors))
            aggregate_metrics['execution_to_goal_errors'].append(row_stats(execution_to_goal_errors))
            if has_plan_to_execution_error:
                aggregate_metrics['final_plan_to_execution_errors'].append(row_stats(final_plan_to_execution_errors))
            aggregate_metrics['plan_to_goal_errors'].append(row_stats(plan_to_goal_errors))
            aggregate_metrics['num_nodes'].append(row_stats(num_nodes))

            print("{:50s}: {:3.2f}% timeout ".format(str(subfolder), timeout_percentage))

    execution_ax.legend()
    planning_ax.legend()

    print('-' * 90)

    for metric_name, table_data in aggregate_metrics.items():
        print(Style.BRIGHT + metric_name + Style.NORMAL)
        table_data_flipped = transpose_2d_lists(table_data)
        table = tabulate(table_data_flipped, headers=headers, tablefmt='github', floatfmt='6.4f')
        print(table)
        print()

    print(Style.BRIGHT + "p-value matrix (goal vs execution)" + Style.NORMAL)
    print(dict_to_pvale_table(execution_to_goal_errors_comparisons, table_format='github'))
    print(Style.BRIGHT + "p-value matrix (plan vs execution)" + Style.NORMAL)
    print(dict_to_pvale_table(plan_to_execution_errors_comparisons, table_format='github'))

    plt.savefig('results/final_tail_error_hist.png')
    if not args.no_plot:
        plt.show()


if __name__ == '__main__':
    main()
