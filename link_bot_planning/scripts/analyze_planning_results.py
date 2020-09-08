#!/usr/bin/env python

import argparse
import gzip
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from colorama import Style, Fore
from tabulate import tabulate

import rospy
from link_bot_planning.results_utils import labeling_params_from_planner_params
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.metric_utils import row_stats, dict_to_pvalue_table


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
        'Final Execution To Goal Error': [],
        'total_time': [],
    }

    with args.analysis_params.open('r') as analysis_params_file:
        analysis_params = json.load(analysis_params_file)
    with args.fallback_labeling_params.open('r') as fallback_labeling_params_file:
        fallback_labeling_params = json.load(fallback_labeling_params_file)

    # The default for where we write results
    first_results_dir = args.results_dirs[0]
    print(f"Writing analysis to {first_results_dir}")

    execution_to_goal_errors_comparisons = {}
    max_error = analysis_params["max_error"]
    errors_thresholds = np.linspace(0.01, max_error, analysis_params["n_error_bins"])
    print('-' * 90)
    if not args.no_plot:
        execution_success_fig, execution_success_ax = plt.subplots(figsize=(16, 10))
        execution_success_ax.set_xlabel("Task Error Threshold")
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

    # For saving metrics since this script is kind of slow
    table_outfile = open(first_results_dir / 'tables.txt', 'w')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    legend_names = []
    for color, subfolder in zip(colors, all_subfolders):
        metrics_filenames = list(subfolder.glob("*_metrics.json.gz"))
        N = len(metrics_filenames)

        with (subfolder / 'metadata.json').open('r') as metadata_file:
            metadata = json.load(metadata_file)
        planner_params = metadata['planner_params']
        labeling_params = labeling_params_from_planner_params(planner_params, fallback_labeling_params)
        goal_threshold = planner_params['goal_threshold']
        scenario = get_scenario(metadata['scenario'])
        table_config = planner_params['table_config']
        nickname = table_config['nickname']
        legend_nickname = " ".join(nickname) if isinstance(nickname, list) else nickname
        legend_names.append(legend_nickname)

        ###############################################################################################

        final_execution_to_goal_errors = []
        total_times = []
        n_recovery = 0
        # TODO: parallelize this
        for plan_idx, metrics_filename in enumerate(metrics_filenames):
            with gzip.open(metrics_filename, 'rb') as metrics_file:
                data_str = metrics_file.read()
            datum = json.loads(data_str.decode("utf-8"))
            total_time = datum['total_time']
            total_times.append(total_time)

            goal = datum['goal']
            final_actual_state = datum['end_state']
            final_execution_to_goal_error = scenario.distance_to_goal(final_actual_state, goal)

            final_execution_to_goal_errors.append(final_execution_to_goal_error)

            steps = datum['steps']
            for step in steps:
                if step['type'] == 'executed_recovery':
                    n_recovery += 1

        ###############################################################################################

        n_for_metrics = len(final_execution_to_goal_errors)
        final_execution_to_goal_errors = np.array(final_execution_to_goal_errors)
        success_percentage = np.count_nonzero(final_execution_to_goal_errors < goal_threshold) / n_for_metrics * 100

        summary_data = {
            'n_examples': N,
            'n_recovery_actions': n_recovery,
            'success_percentage': success_percentage
        }
        print(Fore.GREEN + f"{legend_nickname}" + Fore.RESET)
        print(summary_data)
        table_outfile.write(f"{legend_nickname}")
        table_outfile.write(json.dumps(summary_data, indent=2))

        if not args.no_plot:
            # Execution Success Plot
            execution_successes = []
            for threshold in errors_thresholds:
                success_percentage_at_threshold = np.count_nonzero(
                    final_execution_to_goal_errors < threshold) / n_for_metrics * 100
                execution_successes.append(success_percentage_at_threshold)
            execution_success_ax.plot(errors_thresholds, execution_successes,
                                      label=legend_nickname, linewidth=5, color=color)

        execution_to_goal_errors_comparisons[str(subfolder.name)] = final_execution_to_goal_errors
        headers.append(str(subfolder.name))

        aggregate_metrics['Final Execution To Goal Error'].append(
            make_row(planner_params, final_execution_to_goal_errors, table_format))
        aggregate_metrics['total_time'].append(
            make_row(planner_params, total_times, table_format))

    if not args.no_plot:
        execution_success_ax.axvline(goal_threshold, color='k', linestyle='--')

        execution_success_ax.legend()

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
        table_outfile.write(metric_name)
        table_outfile.write('\n')
        table_outfile.write(table)
        table_outfile.write('\n')
    pvalue_table_title = "p-value matrix (goal vs execution)"
    pvalue_table = dict_to_pvalue_table(execution_to_goal_errors_comparisons, table_format=table_format)
    print(Style.BRIGHT + pvalue_table_title + Style.NORMAL)
    print(pvalue_table)
    table_outfile.write(pvalue_table_title)
    table_outfile.write('\n')
    table_outfile.write(pvalue_table)
    table_outfile.write('\n')
    if not args.no_plot:
        save_unconstrained_layout(execution_success_fig, first_results_dir / "execution_success.png")
        plt.show()


def save_unconstrained_layout(fig, filename, dpi=300):
    fig.set_constrained_layout(False)
    fig.savefig(filename, bbox_inches='tight', dpi=100)


def get_all_subfolders(args):
    all_subfolders = []
    for results_dir in args.results_dirs:
        subfolders = results_dir.iterdir()
        for subfolder in subfolders:
            if subfolder.is_dir():
                all_subfolders.append(subfolder)
    return all_subfolders


def main():
    rospy.init_node("analyse_planning_results")
    np.set_printoptions(suppress=True, precision=4, linewidth=180)
    plt.style.use('paper')

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('analysis_params', type=pathlib.Path)
    parser.add_argument('fallback_labeling_params', type=pathlib.Path)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--final', action='store_true')
    parser.set_defaults(func=metrics_main)

    args = parser.parse_args()

    metrics_main(args)


if __name__ == '__main__':
    main()
