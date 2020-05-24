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

from link_bot_data.classifier_dataset_utils import generate_examples_for_prediction
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.metric_utils import row_stats
from moonshine.moonshine_utils import sequence_of_dicts_to_dict_of_np_arrays


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


def main():
    np.set_printoptions(suppress=True, precision=4, linewidth=180)
    plt.style.use('paper')

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    subparsers = parser.add_subparsers()

    metrics_subparser = subparsers.add_parser('metrics')
    metrics_subparser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    metrics_subparser.add_argument('--no-plot', action='store_true')
    metrics_subparser.add_argument('--final', action='store_true')
    metrics_subparser.add_argument('--ignore-timeouts', action='store_true', help='for error metrics, ignore timeouts/approx sln')
    metrics_subparser.set_defaults(func=metrics_main)

    error_viz_subparser = subparsers.add_parser('error_viz')
    error_viz_subparser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    error_viz_subparser.add_argument('labeling_params', help='labeling params json file', type=pathlib.Path)
    error_viz_subparser.add_argument('--no-plot', action='store_true')
    error_viz_subparser.add_argument('--final', action='store_true')
    error_viz_subparser.set_defaults(func=error_viz_main)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


def error_viz_main(args):
    labeling_params = json.load(args.labeling_params.open("r"))
    all_subfolders = get_all_subfolders(args)
    for subfolder in all_subfolders:
        metrics_filename = subfolder / 'metrics.json'
        metrics = json.load(metrics_filename.open("r"))
        planner_params = metrics['planner_params']
        scenario = get_scenario(planner_params['scenario'])
        table_config = planner_params['table_config']
        nickname = table_config['nickname']
        goal_threshold = planner_params['goal_threshold']

        error_metrics = []
        violations_counts = []
        final_plan_to_execution_errors = []
        data = metrics['metrics']
        num_plans = len(data)
        for traj_idx, datum in enumerate(data):
            planned_path = datum['planned_path']
            actual_path = datum['actual_path']
            final_planned_state = planned_path[-1]
            final_actual_state = actual_path[-1]
            final_plan_to_execution_error = scenario.distance(final_planned_state, final_actual_state)

            # split the planned/actual path into classifier examples
            inputs = datum['environment']
            inputs['traj_idx'] = traj_idx
            planned_path_dict = sequence_of_dicts_to_dict_of_np_arrays(planned_path)
            actual_path_dict = sequence_of_dicts_to_dict_of_np_arrays(actual_path)
            examples_generator = generate_examples_for_prediction(inputs=inputs,
                                                                  outputs=actual_path_dict,
                                                                  predictions=planned_path_dict,
                                                                  start_t=0,
                                                                  labeling_params=labeling_params,
                                                                  prediction_horizon=len(planned_path))

            # count number of not-close (diverged) time steps

            violations = 0
            examples = list(examples_generator)
            for example in examples:
                label = example['label'].numpy().squeeze()
                if not label:
                    # this example violates the MER!
                    violations += 1

            error_metrics_i = (
                final_plan_to_execution_error,
                traj_idx,
                {
                    'violations': violations,
                    'final_error_gt_goal_threshold': final_plan_to_execution_error > goal_threshold,
                }
            )
            violations_counts.append(violations)
            final_plan_to_execution_errors.append(final_plan_to_execution_error)
            error_metrics.append(error_metrics_i)

        violations_counts = np.array(violations_counts)
        final_plan_to_execution_errors = np.array(final_plan_to_execution_errors)

        print(' '.join(nickname))
        print(f"mean final execution to plan error {np.mean(final_plan_to_execution_errors)}")
        print(violations_counts)
        num_plans_with_violations = np.count_nonzero(violations_counts)
        print(f"{num_plans_with_violations}/{num_plans}")
        # sorted_errors_with_indices = sorted(error_metrics, reverse=True)
        # for error, index, other_metrics in sorted_errors_with_indices:
        #     print(f"{index:3d} {error:.4f} {other_metrics['violations']} {other_metrics['final_error_gt_goal_threshold']}")


def metrics_main(args):
    headers = ['']
    aggregate_metrics = {
        'Planning Time': [],
        'Final Execution To Goal Error': [],
        'Final Plan To Goal Error': [],
        'Final Plan To Execution Error': [],
        'Num Nodes': [],
        'Num Steps': [],
    }
    execution_to_goal_errors_comparisons = {}
    plan_to_execution_errors_comparisons = {}
    max_error = 1.25
    errors_thresholds = np.linspace(0.01, max_error, 49)
    print('-' * 90)
    if not args.no_plot:
        plt.figure()
        execution_success_ax = plt.gca()
        execution_success_ax.set_xlabel("Success Threshold, Task Error")
        execution_success_ax.set_ylabel("Success Rate")
        execution_success_ax.set_ylim([-0.1, 100.5])

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
    for color, subfolder in zip(colors, all_subfolders):
        metrics_filename = subfolder / 'metrics.json'
        metrics = json.load(metrics_filename.open("r"))
        planner_params = metrics['planner_params']
        goal_threshold = planner_params['goal_threshold']
        scenario = get_scenario(planner_params['scenario'])
        table_config = planner_params['table_config']
        nickname = table_config['nickname']
        legend_nickname = " ".join(nickname) if isinstance(nickname, list) else nickname
        legend_names.append(legend_nickname)
        data = metrics.pop('metrics')
        N = len(data)
        print("{} has {} examples".format(subfolder, N))

        final_plan_to_execution_errors = []
        final_plan_to_goal_errors = []
        final_execution_to_goal_errors = []
        timeouts = 0
        planning_times = []
        nums_nodes = []
        nums_steps = []
        for plan_idx, datum in enumerate(data):
            planned_path = datum['planned_path']
            actual_path = datum['actual_path']
            final_planned_state = planned_path[-1]
            final_actual_state = actual_path[-1]
            final_plan_to_goal_error = scenario.distance_to_goal(final_planned_state, datum['goal'])
            final_execution_to_goal_error = scenario.distance_to_goal(final_actual_state, datum['goal'])
            final_plan_to_execution_error = scenario.distance(final_planned_state, final_actual_state)

            if datum['planner_status'] != "Exact solution":
                timeouts += 1
                # Do not plot metrics for plans which timeout.
                if args.ignore_timeouts:
                    continue

            final_plan_to_execution_errors.append(final_plan_to_execution_error)
            final_plan_to_goal_errors.append(final_plan_to_goal_error)
            final_execution_to_goal_errors.append(final_execution_to_goal_error)

            num_nodes = datum['num_nodes']
            nums_nodes.append(num_nodes)

            num_steps = len(planned_path)
            nums_steps.append(num_steps)

            planning_times.append(datum['planning_time'])

        timeout_percentage = timeouts / N * 100
        n_exact_solutions = N - timeouts
        percentage_solved = n_exact_solutions / N * 100
        percentages_solved.append(percentage_solved)
        n_for_metrics = n_exact_solutions if args.ignore_timeouts else N

        if not args.no_plot:
            # Execution Success Plot
            execution_successes = []
            for threshold in errors_thresholds:
                success_percentage = np.count_nonzero(final_execution_to_goal_errors < threshold) / n_for_metrics * 100
                execution_successes.append(success_percentage)
            execution_success_ax.plot(errors_thresholds, execution_successes, label=legend_nickname, linewidth=5, color=color)

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
        aggregate_metrics['Final Plan To Goal Error'].append(make_row(planner_params, final_plan_to_goal_errors, table_format))
        aggregate_metrics['Final Execution To Goal Error'].append(
            make_row(planner_params, final_execution_to_goal_errors, table_format))
        aggregate_metrics['Num Nodes'].append(make_row(planner_params, nums_nodes, table_format))
        aggregate_metrics['Num Steps'].append(make_row(planner_params, nums_steps, table_format))

        print("{:50s}: {:3.2f}% timeout ".format(str(subfolder), timeout_percentage))
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
        timeout_ax = plt.gca()
        timeout_ax.bar(legend_names, percentages_solved, color=colors)
        timeout_ax.set_ylabel("percentage solved")
        timeout_ax.set_title("Percentage Solved")
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
    print(dict_to_pvale_table(execution_to_goal_errors_comparisons, table_format=table_format))
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


if __name__ == '__main__':
    main()
