#!/usr/bin/env python

import argparse
import json
import pathlib
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from colorama import Style, Fore
from scipy import stats
from tabulate import tabulate

from link_bot_planning.get_scenario import get_scenario
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.metric_utils import breif_row_stats


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


def make_cell(text, tablefmt):
    if isinstance(text, list):
        if tablefmt == 'latex_raw':
            return "\\makecell{" + "\\\\".join(text) + "}"
        else:
            return "\n".join(text)
    else:
        return text


def make_row(planner_params, metric_data, tablefmt):
    table_config = planner_params['table_config']
    row = [
        make_cell(table_config["nickname"], tablefmt),
        make_cell(table_config["dynamics"], tablefmt),
        make_cell(table_config["classifier"], tablefmt),
    ]
    row.extend(breif_row_stats(metric_data))
    return row


def main():
    plt.style.use('paper')

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_dirs', help='folders containing folders containing metrics.json', type=pathlib.Path, nargs='+')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--final', action='store_true')

    args = parser.parse_args()

    headers = ['']
    aggregate_metrics = {
        'Planning Time': [],
        'Final Execution To Goal Error': [],
        'Final Plan To Goal Error': [],
        'Final Plan To Execution Error': [],
        'Num Nodes': [],
    }

    execution_to_goal_errors_comparisons = {}
    plan_to_execution_errors_comparisons = {}
    max_error = 1.5
    errors_thresholds = np.linspace(0.01, max_error, 49)
    print('-' * 90)
    if not args.no_plot:
        plt.figure()
        execution_success_ax = plt.gca()
        execution_success_ax.set_xlabel("Success Threshold, Task Error")
        execution_success_ax.set_ylabel("Success Rate")
        execution_success_ax.set_ylim([-0.1, 100.1])

        plt.figure()
        planning_success_ax = plt.gca()
        planning_success_ax.set_xlabel("Success Threshold, Task Error")
        planning_success_ax.set_ylabel("Success Rate")
        planning_success_ax.set_ylim([-0.1, 100.1])

        plt.figure()
        execution_error_ax = plt.gca()
        execution_error_ax.set_xlabel("Task Error")
        execution_error_ax.set_ylabel("Density")

    all_subfolders = []
    for results_dir in args.results_dirs:
        subfolders = results_dir.iterdir()
        for subfolder in subfolders:
            if subfolder.is_dir():
                all_subfolders.append(subfolder)

    if args.final:
        table_format = 'latex_raw'
        for subfolder_idx, subfolder in enumerate(all_subfolders):
            print("{}) {}".format(subfolder_idx, subfolder))
        sort_order = input(Fore.CYAN + "Enter the desired table order:\n" + Fore.RESET)
        all_subfolders = [all_subfolders[int(i)] for i in sort_order.split(' ')]
    else:
        table_format = 'fancy_grid'

    max_density = 0
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for color, subfolder in zip(colors, all_subfolders):
        metrics_filename = subfolder / 'metrics.json'
        metrics = json.load(metrics_filename.open("r"))
        planner_params = metrics['planner_params']
        goal_threshold = planner_params['goal_threshold']
        scenario = get_scenario(planner_params['scenario'])
        timeout = planner_params['timeout']
        table_config = planner_params['table_config']
        nickname = table_config['nickname']
        nickname = "".join(nickname) if isinstance(nickname, list) else nickname
        data = metrics.pop('metrics')
        N = len(data)
        print("{} has {} examples".format(subfolder, N))

        final_plan_to_execution_errors = []
        final_plan_to_goal_errors = []
        final_execution_to_goal_errors = []
        timeouts = 0
        planning_times = []
        nums_nodes = []
        for datum in data:
            planned_path = datum['planned_path']
            actual_path = datum['actual_path']
            final_planned_state = planned_path[-1]
            final_actual_state = actual_path[-1]
            final_plan_to_goal_error = scenario.distance_to_goal(final_planned_state, datum['goal'])
            final_execution_to_goal_error = scenario.distance_to_goal(final_actual_state, datum['goal'])
            final_plan_to_execution_error = scenario.distance(final_planned_state, final_actual_state)
            final_plan_to_execution_errors.append(final_plan_to_execution_error)
            final_plan_to_goal_errors.append(final_plan_to_goal_error)
            final_execution_to_goal_errors.append(final_execution_to_goal_error)

            num_nodes = datum['num_nodes']
            nums_nodes.append(num_nodes)

            planning_times.append(datum['planning_time'])

            if datum['planning_time'] > timeout:
                timeouts += 1

        timeout_percentage = timeouts / N * 100

        if not args.no_plot:
            # Execution Success Plot
            execution_successes = []
            for threshold in errors_thresholds:
                success_percentage = np.count_nonzero(final_execution_to_goal_errors < threshold) / N * 100
                execution_successes.append(success_percentage)
            execution_success_ax.plot(errors_thresholds, execution_successes, label=nickname, linewidth=5, color=color)

            # Execution Error Plot
            final_execution_to_goal_pdf = stats.gaussian_kde(final_execution_to_goal_errors)
            final_execution_to_goal_densities_at_thresholds = final_execution_to_goal_pdf(errors_thresholds)
            execution_error_ax.plot(errors_thresholds, final_execution_to_goal_densities_at_thresholds, label=nickname,
                                    linewidth=5,
                                    c=color)
            max_density = max(np.max(final_execution_to_goal_densities_at_thresholds), max_density)

            # Planning SuccessPlot
            planning_successes = []
            for threshold in errors_thresholds:
                success_percentage = np.count_nonzero(final_plan_to_execution_errors < threshold) / N * 100
                planning_successes.append(success_percentage)
            planning_success_ax.plot(errors_thresholds, planning_successes, label=nickname, linewidth=5, c=color)

        execution_to_goal_errors_comparisons[str(subfolder.name)] = final_execution_to_goal_errors
        plan_to_execution_errors_comparisons[str(subfolder.name)] = final_plan_to_execution_errors
        headers.append(str(subfolder.name))

        aggregate_metrics['Planning Time'].append(make_row(planner_params, planning_times, table_format))
        aggregate_metrics['Final Plan To Execution Error'].append(
            make_row(planner_params, final_plan_to_execution_errors, table_format))
        aggregate_metrics['Final Plan To Goal Error'].append(make_row(planner_params, final_plan_to_goal_errors, table_format))
        aggregate_metrics['Final Execution To Goal Error'].append(
            make_row(planner_params, final_execution_to_goal_errors, table_format))
        aggregate_metrics['Num Nodes'].append(make_row(planner_params, nums_nodes, table_format))

        print("{:50s}: {:3.2f}% timeout ".format(str(subfolder), timeout_percentage))

    if not args.no_plot:
        execution_success_ax.plot([goal_threshold, goal_threshold], [0, 100], color='k', linestyle='--')
        execution_error_ax.plot([goal_threshold, goal_threshold], [0, max_density], color='k', linestyle='--')
        planning_success_ax.plot([goal_threshold, goal_threshold], [0, 100], color='k', linestyle='--')

        execution_success_ax.set_title("Success In Execution, {}".format(scenario))
        planning_success_ax.set_title("Success In Planning, {}".format(scenario))
        execution_error_ax.set_title("Task Error, {}".format(scenario))
        execution_success_ax.legend()
        execution_error_ax.legend()
        planning_success_ax.legend()

    print('-' * 90)

    for metric_name, table_data in aggregate_metrics.items():
        print(Style.BRIGHT + metric_name + Style.NORMAL)
        table = tabulate(table_data, tablefmt=table_format, floatfmt='6.4f', numalign='center', stralign='left')
        print(table)
        print()

    print(Style.BRIGHT + "p-value matrix (goal vs execution)" + Style.NORMAL)
    print(dict_to_pvale_table(execution_to_goal_errors_comparisons, table_format=table_format))

    if not args.no_plot:
        plt.show()


if __name__ == '__main__':
    main()
