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

    with (args.results_dir / 'metadata.json').open('r') as metadata_file:
        metadata_str = metadata_file.read()
        metadata = json.loads(metadata_str)
    scenario = get_scenario(metadata['scenario'])

    for plan_idx in args.plan_idx:
        with gzip.open(args.results_dir / f'{plan_idx}_metrics.json.gz', 'rb') as metrics_file:
            metrics_str = metrics_file.read()
        metrics = json.loads(metrics_str.decode("utf-8"))
        plot_plan(scenario, metrics, plan_idx, metadata)


def plot_plan(scenario, metrics_for_plan, plan_idx, metadata):
    planner_params = metadata['planner_params']
    labeling_params = labeling_params_from_planner_params(planner_params)
    goal = metrics_for_plan['goal']
    environment = numpify(metrics_for_plan['environment'])
    planned_path = metrics_for_plan['planned_path']
    actual_path = metrics_for_plan['actual_path']

    planned_actions = metrics_for_plan['actions']

    scenario.animate_evaluation_results(environment=environment,
                                        actual_states=actual_path,
                                        predicted_states=planned_path,
                                        actions=planned_actions,
                                        goal=goal,
                                        goal_threshold=planner_params['goal_threshold'],
                                        labeling_params=labeling_params,
                                        accept_probabilities=None,
                                        horizon=metadata['horizon'])


if __name__ == '__main__':
    main()
