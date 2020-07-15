#!/usr/bin/env python

import argparse
import json
import rospy
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
    np.set_printoptions(suppress=True, precision=4, linewidth=180)
    plt.style.use('paper')

    parser = argparse.ArgumentParser(formatter_class=my_formatter)

    parser.add_argument('results_dirs', help='results directory', type=pathlib.Path, nargs='+')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--final', action='store_true')

    args = parser.parse_args()

    all_subfolders = get_all_subfolders(args)

    if args.final:
        table_format = 'latex_raw'
        for subfolder_idx, subfolder in enumerate(all_subfolders):
            print("{}) {}".format(subfolder_idx, subfolder))
        sort_order = input(Fore.CYAN + "Enter the desired table order:\n" + Fore.RESET)
        all_subfolders = [all_subfolders[int(i)] for i in sort_order.split(' ')]
    else:
        table_format = 'fancy_grid'

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    legend_names = []
    total_recovery_actions_takens = []
    for color, subfolder in zip(colors, all_subfolders):
        metrics_filenames = list(subfolder.glob("*_metrics.json.gz"))
        N = len(metrics_filenames)
        print(Fore.GREEN + f"{subfolder} has {N} examples" + Fore.RESET)

        with (subfolder / 'metadata.json').open('r') as metadata_file:
            metadata = json.load(metadata_file)
        planner_params = metadata['planner_params']
        scenario = get_scenario(metadata['scenario'])
        table_config = planner_params['table_config']
        nickname = table_config['nickname']
        legend_nickname = " ".join(nickname) if isinstance(nickname, list) else nickname
        legend_names.append(legend_nickname)

        # TODO: parallelize this
        total_recovery_actions_taken = 0
        for plan_idx, metrics_filename in enumerate(metrics_filenames):
            with gzip.open(metrics_filename, 'rb') as metrics_file:
                data_str = metrics_file.read()
            datum = json.loads(data_str.decode("utf-8"))
            recovery_actions_result = datum['recovery_actions_result']
            recovery_actions_taken = recovery_actions_result['recovery_actions_taken']
            n_recovery_actions_taken = len(recovery_actions_taken)
            n_attempts = datum['planner_params']['recovery']['n_attempts']

            if n_recovery_actions_taken > 0:
                print(
                    f"on plan {plan_idx}, method {legend_nickname} executed {n_recovery_actions_taken}/{n_attempts} recovery actions")
                total_recovery_actions_taken += n_recovery_actions_taken
        total_recovery_actions_takens.append(total_recovery_actions_taken)

    print(legend_names, total_recovery_actions_takens)
    plt.bar(legend_names, total_recovery_actions_takens)
    plt.show()


if __name__ == '__main__':
    main()
