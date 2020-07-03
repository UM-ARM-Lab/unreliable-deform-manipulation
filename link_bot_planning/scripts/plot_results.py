#!/usr/bin/env python

import argparse
import gzip
import rospy
import json
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
    parser.add_argument("plan_idx", type=int_range_arg, help='which plan to show')
    parser.add_argument("--save", action='store_true')

    rospy.init_node("plot_results")

    args = parser.parse_args()

    metrics_filename = args.results_dir / "metrics.json.gz"
    with gzip.open(metrics_filename, 'rb') as metrics_file:
        data_str = metrics_file.read()
        data = json.loads(data_str.decode("utf-8"))
    scenario = get_scenario(data['scenario'])
    planner_params = data['planner_params']
    labeling_params = labeling_params_from_planner_params(planner_params)

    metrics = data['metrics']

    for plan_idx in args.plan_idx:
        plot_plan(scenario, metrics, plan_idx, labeling_params, planner_params)


def plot_plan(scenario, metrics, plan_idx, labeling_params, planner_params):
    metric_for_plan = metrics[plan_idx]
    goal = metric_for_plan['goal']
    environment = numpify(metric_for_plan['environment'])
    planned_path = metric_for_plan['planned_path']
    actual_path = metric_for_plan['actual_path']

    planned_actions = metric_for_plan['actions']

    if labeling_params is not None:
        is_close = np.linalg.norm(p - a, axis=1) < labeling_params['threshold']
    else:
        is_close = None

    scenario.animate_evaluation_results(environment=environment,
                                        actual_states=actual_path,
                                        predicted_states=planned_path,
                                        actions=planned_actions,
                                        labels=is_close,
                                        goal=goal,
                                        goal_threshold=planner_params['goal_threshold'],
                                        accept_probabilities=None)


if __name__ == '__main__':
    main()
