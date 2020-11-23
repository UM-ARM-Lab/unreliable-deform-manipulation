#!/usr/bin/env python
import argparse
import gzip
import json
import pathlib
import pickle
from typing import List

import colorama
import hjson
import matplotlib.pyplot as plt
import numpy as np
import orjson
from colorama import Style, Fore
from tabulate import tabulate

import rospy
from arc_utilities.filesystem_utils import get_all_subfolders
from link_bot_planning.results_metrics import FinalExecutionToGoalError, NRecoveryActions, NPlanningAttempts, TotalTime, \
    ResultsMetric
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.metric_utils import dict_to_pvalue_table


def metrics_main(args):
    with args.analysis_params.open('r') as analysis_params_file:
        analysis_params = hjson.load(analysis_params_file)

    # The default for where we write results
    out_dir = args.results_dirs[0]
    print(f"Writing analysis to {out_dir}")

    # For saving metrics since this script is kind of slow
    table_outfile = open(out_dir / 'tables.txt', 'w')

    subfolders = get_all_subfolders(args)

    if args.final:
        table_format = 'latex_raw'
        for subfolder_idx, subfolder in enumerate(subfolders):
            print("{}) {}".format(subfolder_idx, subfolder))
        sort_order = input(Fore.CYAN + "Enter the desired table order:\n" + Fore.RESET)
        subfolders_ordered = [subfolders[int(i)] for i in sort_order.split(' ')]
    else:
        table_format = 'fancy_grid'
        subfolders_ordered = subfolders

    legend_names = []

    pickle_filename = out_dir / "metrics.pkl"
    if pickle_filename.exists():
        rospy.loginfo(Fore.GREEN + f"Loading existing metrics from {pickle_filename}")
        with pickle_filename.open("rb") as pickle_file:
            metrics: List[ResultsMetric] = pickle.load(pickle_file)

        sort_order_dict = {}
        for sort_idx, subfolder in enumerate(subfolders_ordered):
            with (subfolder / 'metadata.json').open('r') as metadata_file:
                metadata = json.load(metadata_file)
            method_name = metadata['planner_params']['method_name']
            sort_order_dict[method_name] = sort_idx

        for metric in metrics:
            metric.sort_methods(sort_order_dict)
    else:
        rospy.loginfo(Fore.GREEN + f"Generating metrics")
        metrics = generate_metrics(analysis_params, args, legend_names, out_dir, subfolders_ordered)

        with pickle_filename.open("wb") as pickle_file:
            pickle.dump(metrics, pickle_file)
        rospy.loginfo(Fore.GREEN + f"Pickling metrics to {pickle_filename}")

    for metric in metrics:
        metric.enumerate_methods()

    for metric in metrics:
        table_header, table_data = metric.make_table(table_format)
        print(Style.BRIGHT + metric.name + Style.NORMAL)
        table = tabulate(table_data,
                         headers=table_header,
                         tablefmt=table_format,
                         floatfmt='6.4f',
                         numalign='center',
                         stralign='left')
        print(table)
        print()
        table_outfile.write(metric.name)
        table_outfile.write('\n')
        table_outfile.write(table)
        table_outfile.write('\n')

    for metric in metrics:
        pvalue_table_title = f"p-value matrix [{metric.name}]"
        pvalue_table = dict_to_pvalue_table(metric.values, table_format=table_format)
        print(Style.BRIGHT + pvalue_table_title + Style.NORMAL)
        print(pvalue_table)
        table_outfile.write(pvalue_table_title)
        table_outfile.write('\n')
        table_outfile.write(pvalue_table)
        table_outfile.write('\n')

    for metric in metrics:
        metric.make_figure()
        metric.finish_figure()
        metric.save_figure()

    if not args.no_plot:
        plt.show()


def generate_metrics(analysis_params, args, legend_names, out_dir, subfolders_ordered):
    metrics = [
        FinalExecutionToGoalError(args, analysis_params, out_dir),
        NRecoveryActions(args, analysis_params, out_dir),
        TotalTime(args, analysis_params, out_dir),
        NPlanningAttempts(args, analysis_params, out_dir),
    ]
    for subfolder in subfolders_ordered:
        metrics_filenames = list(subfolder.glob("*_metrics.json.gz"))

        with (subfolder / 'metadata.json').open('r') as metadata_file:
            metadata_str = metadata_file.read()
        metadata = json.loads(metadata_str)
        method_name = metadata['planner_params']['method_name']
        scenario = get_scenario(metadata['scenario'])
        legend_method_name = ""  # FIXME: how to specify this?
        legend_names.append(legend_method_name)

        for metric in metrics:
            metric.setup_method(method_name, metadata)

        datums = []
        for plan_idx, metrics_filename in enumerate(metrics_filenames):
            if args.debug and plan_idx > 3:
                break
            with gzip.open(metrics_filename, 'rb') as metrics_file:
                data_str = metrics_file.read()
            # orjson is twice as fast, and yes it really matters here.
            datum = orjson.loads(data_str.decode("utf-8"))
            datums.append(datum)

        # NOTE: even though this is slow, parallelizing is not easy because "scenario" cannot be pickled
        for metric in metrics:
            for datum in datums:
                metric.aggregate_trial(method_name, scenario, datum)

        for metric in metrics:
            metric.convert_to_numpy_arrays()
    return metrics


def main():
    colorama.init(autoreset=True)

    rospy.init_node("analyse_planning_results")
    np.set_printoptions(suppress=True, precision=4, linewidth=180)
    plt.style.use('paper')

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('analysis_params', type=pathlib.Path)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--debug', action='store_true', help='will only run on a few examples to speed up debugging')
    parser.set_defaults(func=metrics_main)

    args = parser.parse_args()

    metrics_main(args)


if __name__ == '__main__':
    main()
