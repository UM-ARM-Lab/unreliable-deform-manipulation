#!/usr/bin/env python

import argparse
import json
import pathlib
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import gzip
from colorama import Style, Fore
from scipy import stats
from tabulate import tabulate

from link_bot_data.classifier_dataset_utils import generate_classifier_examples_from_batch
from link_bot_planning.results_utils import labeling_params_from_planner_params
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.metric_utils import row_stats, dict_to_pvalue_table
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_np_arrays


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


def make_header():
    return ["Name", "Dynamics", "Classifier", "min", "max", "mean", "median", "std"]


def make_row(planner_params, metric_data, tablefmt):
    table_config = planner_params['table_config']
    row = [
        make_cell(table_config["nickname"], tablefmt),
        make_cell(table_config["dynamics"], tablefmt),
        make_cell(table_config["classifier"], tablefmt),
    ]
    row.extend(row_stats(metric_data))
    return row


def metrics_main(args):
    headers = ['']
    aggregate_metrics = {
        'Planning Time': [],
        'Final Execution To Goal Error': [],
        'Final Plan To Goal Error': [],
        'Final Plan To Execution Error': [],
        'Num Nodes': [],
        'Num Steps': [],
        '% Steps with MER Violations': [],
    }

    with args.analysis_params.open('r') as analysis_params_file:
        analysis_params = json.load(analysis_params_file)

    execution_to_goal_errors_comparisons = {}
    plan_to_execution_errors_comparisons = {}
    max_error = analysis_params["max_error"]
    errors_thresholds = np.linspace(0.01, max_error, analysis_params["n_error_bins"])
    print('-' * 90)
    if not args.no_plot:
        plt.figure()
        planning_success_ax = plt.gca()
        planning_success_ax.set_xlabel("Success Threshold, Task Error")
        planning_success_ax.set_ylabel("Success Rate")
        planning_success_ax.set_ylim([-0.1, 100.5])

        plt.figure()
        execution_error_ax = plt.gca()
        execution_error_ax.set_xlabel("Task Error")
        execution_error_ax.set_ylabel("Density")

        plt.figure()
        planning_error_ax = plt.gca()
        planning_error_ax.set_xlabel("Task Error")
        planning_error_ax.set_ylabel("Density")

        plt.figure()
        execution_success_ax = plt.gca()
        execution_success_ax.set_xlabel("Success Threshold, Task Error")
        execution_success_ax.set_ylabel("Success Rate")
        execution_success_ax.set_ylim([-0.1, 100.5])

    all_subfolders = get_all_subfolders(args)

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
    legend_names = []
    percentages_solved = []
    percentages_timeout = []
    percentages_not_progressing = []
    for color, subfolder in zip(colors, all_subfolders):
        metrics_filenames = list(subfolder.glob("*_metrics.json.gz"))
        N = len(metrics_filenames)
        print(Fore.GREEN + f"{subfolder} has {N} examples" + Fore.RESET)

        final_plan_to_execution_errors = []
        final_plan_to_goal_errors = []
        final_execution_to_goal_errors = []
        solveds = 0
        timeouts = 0
        not_progressings = 0
        planning_times = []
        nums_nodes = []
        nums_steps = []
        nums_mer_violations = []

        with (subfolder / 'metadata.json').open('r') as metadata_file:
            metadata = json.load(metadata_file)
        planner_params = metadata['planner_params']
        labeling_params = labeling_params_from_planner_params(planner_params, fallback_labeling_params=None)
        goal_threshold = planner_params['goal_threshold']
        scenario = get_scenario(metadata['scenario'])
        table_config = planner_params['table_config']
        nickname = table_config['nickname']
        legend_nickname = " ".join(nickname) if isinstance(nickname, list) else nickname
        legend_names.append(legend_nickname)

        for plan_idx, metrics_filename in enumerate(metrics_filenames):
            with gzip.open(metrics_filename, 'rb') as metrics_file:
                data_str = metrics_file.read()
            datum = json.loads(data_str.decode("utf-8"))

            planned_path = datum['planned_path']
            actual_path = datum['actual_path']
            final_planned_state = planned_path[-1]
            final_actual_state = actual_path[-1]
            final_plan_to_goal_error = scenario.distance_to_goal(final_planned_state, datum['goal'])
            final_execution_to_goal_error = scenario.distance_to_goal(final_actual_state, datum['goal'])
            final_plan_to_execution_error = scenario.distance(final_planned_state, final_actual_state)
            p = sequence_of_dicts_to_dict_of_np_arrays(planned_path)['link_bot']
            a = sequence_of_dicts_to_dict_of_np_arrays(actual_path)['link_bot']

            if datum['planner_status'] == "solved":
                solveds += 1
                final_plan_to_execution_errors.append(final_plan_to_execution_error)
                final_plan_to_goal_errors.append(final_plan_to_goal_error)
                final_execution_to_goal_errors.append(final_execution_to_goal_error)

                num_nodes = datum['num_nodes']
                nums_nodes.append(num_nodes)

                num_steps = len(planned_path)
                nums_steps.append(num_steps)

                if labeling_params is not None:
                    is_close = np.linalg.norm(p - a, axis=1) < labeling_params['threshold']
                    num_mer_violations = np.count_nonzero(1 - is_close) / num_steps * 100
                else:
                    num_mer_violations = 0
                nums_mer_violations.append(num_mer_violations)

                planning_times.append(datum['planning_time'])
            elif datum['planner_status'] == "not progressing":
                not_progressings += 1
                continue
            elif datum['planner_status'] == "timeout":
                timeouts += 1
                continue
            else:
                raise NotImplementedError()

        percentage_solved = solveds / N * 100
        percentages_solved.append(percentage_solved)
        percentage_not_progressing = not_progressings / N * 100
        percentages_not_progressing.append(percentage_not_progressing)
        percentage_timeout = timeouts / N * 100
        percentages_timeout.append(percentage_timeout)
        n_for_metrics = solveds

        if not args.no_plot:
            # Execution Success Plot
            execution_successes = []
            for threshold in errors_thresholds:
                success_percentage = np.count_nonzero(final_execution_to_goal_errors < threshold) / n_for_metrics * 100
                execution_successes.append(success_percentage)
            execution_success_ax.plot(errors_thresholds, execution_successes,
                                      label=legend_nickname, linewidth=5, color=color)

            # Execution Error Plot
            final_execution_to_goal_pdf = stats.gaussian_kde(final_execution_to_goal_errors)
            final_execution_to_goal_densities_at_thresholds = final_execution_to_goal_pdf(errors_thresholds)
            execution_error_ax.plot(errors_thresholds, final_execution_to_goal_densities_at_thresholds, label=legend_nickname,
                                    linewidth=5,
                                    c=color)
            max_density = max(np.max(final_execution_to_goal_densities_at_thresholds), max_density)

            # Planning Success Plot
            planning_successes = []
            for threshold in errors_thresholds:
                success_percentage = np.count_nonzero(final_plan_to_execution_errors < threshold) / n_for_metrics * 100
                planning_successes.append(success_percentage)
            planning_success_ax.plot(errors_thresholds, planning_successes, label=legend_nickname, linewidth=5, c=color)

            # Planning Error Plot
            final_planning_to_goal_pdf = stats.gaussian_kde(final_plan_to_execution_errors)
            final_planning_to_goal_densities_at_thresholds = final_planning_to_goal_pdf(errors_thresholds)
            planning_error_ax.plot(errors_thresholds, final_planning_to_goal_densities_at_thresholds, label=legend_nickname,
                                   linewidth=5,
                                   c=color)
            max_density = max(np.max(final_planning_to_goal_densities_at_thresholds), max_density)

        execution_to_goal_errors_comparisons[str(subfolder.name)] = final_execution_to_goal_errors
        plan_to_execution_errors_comparisons[str(subfolder.name)] = final_plan_to_execution_errors
        headers.append(str(subfolder.name))

        aggregate_metrics['Planning Time'].append(make_row(planner_params, planning_times, table_format))
        aggregate_metrics['Final Plan To Execution Error'].append(
            make_row(planner_params, final_plan_to_execution_errors, table_format))
        aggregate_metrics['Final Plan To Goal Error'].append(
            make_row(planner_params, final_plan_to_goal_errors, table_format))
        aggregate_metrics['Final Execution To Goal Error'].append(
            make_row(planner_params, final_execution_to_goal_errors, table_format))
        aggregate_metrics['Num Nodes'].append(make_row(planner_params, nums_nodes, table_format))
        aggregate_metrics['Num Steps'].append(make_row(planner_params, nums_steps, table_format))
        aggregate_metrics['% Steps with MER Violations'].append(
            make_row(planner_params, nums_mer_violations, table_format))

        print(f"{subfolder.name:30s}: {percentage_timeout:3.2f}% timeout")
        for error, plan_idx in sorted(zip(final_execution_to_goal_errors, range(len(final_execution_to_goal_errors)))):
            print(f"{plan_idx}: {error:5.3f} error between execution to goal")
        if labeling_params is not None:
            for num_mer_violations, plan_idx in sorted(zip(nums_mer_violations, range(len(nums_mer_violations)))):
                print(f"{plan_idx}: {num_mer_violations:5.1f}% of steps violate MER")
    if not args.no_plot:
        execution_success_ax.plot([goal_threshold, goal_threshold], [0, 100], color='k', linestyle='--')
        execution_error_ax.plot([goal_threshold, goal_threshold], [0, max_density], color='k', linestyle='--')
        planning_success_ax.plot([goal_threshold, goal_threshold], [0, 100], color='k', linestyle='--')

        execution_success_ax.set_title("Success In Execution, {}".format(scenario))
        planning_success_ax.set_title("Success In Planning, {}".format(scenario))
        execution_error_ax.set_title("Execution Task Error, {}".format(scenario))
        planning_error_ax.set_title("Planning Task Error, {}".format(scenario))
        execution_success_ax.legend()
        execution_error_ax.legend()
        planning_success_ax.legend()
        planning_error_ax.legend()

        # Timeout Plot
        plt.figure()
        percentage_ax = plt.gca()
        timeout_bar = percentage_ax.bar(legend_names, percentages_timeout)
        not_progressing_bar = percentage_ax.bar(legend_names, percentages_not_progressing, bottom=percentages_timeout)
        solved_bar = percentage_ax.bar(legend_names, percentages_solved,
                                       bottom=np.add(percentages_not_progressing, percentages_timeout))
        percentage_ax.set_ylabel("Percentage")
        percentage_ax.set_xlabel("Methods")
        percentage_ax.set_title("Planner Status Breakdown")
        percentage_ax.legend((solved_bar, not_progressing_bar,  timeout_bar),
                             ("Solved", "NotProgressing", "Timeout"))
    print('-' * 90)

    for metric_name, table_data in aggregate_metrics.items():
        print(Style.BRIGHT + metric_name + Style.NORMAL)
        table = tabulate(table_data,
                         headers=make_header(),
                         tablefmt=table_format,
                         floatfmt='6.4f',
                         numalign='center',
                         stralign='left')
        print(table)
        print()
    print(Style.BRIGHT + "p-value matrix (goal vs execution)" + Style.NORMAL)
    print(dict_to_pvalue_table(execution_to_goal_errors_comparisons, table_format=table_format))
    if not args.no_plot:
        plt.show()


def get_all_subfolders(args):
    all_subfolders = []
    for results_dir in args.results_dirs:
        subfolders = results_dir.iterdir()
        for subfolder in subfolders:
            if subfolder.is_dir():
                all_subfolders.append(subfolder)
    return all_subfolders


def main():
    np.set_printoptions(suppress=True, precision=4, linewidth=180)
    plt.style.use('paper')

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    subparsers = parser.add_subparsers()

    metrics_subparser = subparsers.add_parser('metrics')
    metrics_subparser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    metrics_subparser.add_argument('analysis_params', type=pathlib.Path)
    metrics_subparser.add_argument('--no-plot', action='store_true')
    metrics_subparser.add_argument('--final', action='store_true')
    metrics_subparser.set_defaults(func=metrics_main)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
