#!/usr/bin/env python
from time import sleep

import argparse
import matplotlib.pyplot as plt
import gzip
import rospy
import json
from typing import Dict
import pathlib

import numpy as np

from link_bot_planning.results_utils import labeling_params_from_planner_params
from link_bot_pycommon.args import my_formatter, int_range_arg
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.moonshine_utils import numpify, sequence_of_dicts_to_dict_of_np_arrays


def main():
    np.set_printoptions(linewidth=250, precision=3, suppress=True)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')

    rospy.init_node("plot_results")

    args = parser.parse_args()

    with (args.results_dir / 'metadata.json').open('r') as metadata_file:
        metadata_str = metadata_file.read()
        metadata = json.loads(metadata_str)

    metrics_filenames = list(args.results_dir.glob("*_metrics.json.gz"))
    N = len(metrics_filenames)
    planning_times = []
    planning_times_by_status = {}
    timeout = metadata['planner_params']["termination_criteria"]['timeout']
    N = len(metrics_filenames)
    n_timeouts = 0
    for metrics_filename in metrics_filenames:
        with gzip.open(metrics_filename) as metrics_file:
            metrics_str = metrics_file.read()
        metrics = json.loads(metrics_str.decode("utf-8"))

        planner_status = metrics['planner_status']
        planning_time = metrics['planning_time']
        if planner_status == "timeout":
            n_timeouts += 1
        print(planner_status, planning_time)
        planning_times.append(planning_time)
        if planner_status not in planning_times_by_status:
            planning_times_by_status[planner_status] = []
        planning_times_by_status[planner_status].append(planning_time)

    plt.style.use("slides")

    print(f"{n_timeouts / N * 100}% timeout")

    plt.figure()
    plt.title("planning times")
    plt.hist(planning_times, bins=100)
    plt.xlabel("planning time (s)")
    plt.ylabel("count")
    plt.axvline(timeout, color='r')

    for status, times in planning_times_by_status.items():
        plt.figure()
        plt.title(f"planning times [{status}]")
        plt.hist(times, bins=100)
        plt.xlabel("planning time (s)")
        plt.ylabel("count")
        plt.axvline(timeout, color='r')

    plt.show()


if __name__ == '__main__':
    main()
