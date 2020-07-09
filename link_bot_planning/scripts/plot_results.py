#!/usr/bin/env python
from time import sleep

import argparse
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
    parser.add_argument("plan_idx", type=int_range_arg, help='which plan to show')
    parser.add_argument("fallback_labeling_params", type=pathlib.Path,
                        help='labeling params to use in case none can be found automatically')
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--filter-by-status", type=str, nargs="+")
    parser.add_argument("--show-tree", action="store_true")

    rospy.init_node("plot_results")

    args = parser.parse_args()

    with (args.results_dir / 'metadata.json').open('r') as metadata_file:
        metadata_str = metadata_file.read()
        metadata = json.loads(metadata_str)
    scenario = get_scenario(metadata['scenario'])

    with args.fallback_labeling_params.open("r") as fallback_labeling_params_file:
        fallback_labeing_params = json.load(fallback_labeling_params_file)

    for plan_idx in args.plan_idx:
        with gzip.open(args.results_dir / f'{plan_idx}_metrics.json.gz', 'rb') as metrics_file:
            metrics_str = metrics_file.read()
        metrics = json.loads(metrics_str.decode("utf-8"))

        planner_status = metrics['planner_status']
        if args.filter_by_status:
            if planner_status in args.filter_by_status:
                print(plan_idx)
                plot_plan(args, scenario, metrics, plan_idx, metadata, fallback_labeing_params)
        else:
            plot_plan(args, scenario, metrics, plan_idx, metadata, fallback_labeing_params)


def plot_plan(args, scenario, metrics_for_plan, plan_idx, metadata, fallback_labeing_params: Dict):
    planner_params = metadata['planner_params']
    labeling_params = labeling_params_from_planner_params(planner_params, fallback_labeing_params)
    goal = metrics_for_plan['goal']
    environment = numpify(metrics_for_plan['environment'])
    planned_path = metrics_for_plan['planned_path']
    actual_path = metrics_for_plan['actual_path']

    planned_actions = metrics_for_plan['actions']

    scenario.reset_planning_viz()
    if args.show_tree:
        for vertex in metrics_for_plan['tree_json']['vertices']:
            scenario.plot_tree_state(vertex, color='#77777722')
            sleep(0.01)
        # for edge in metrics_for_plan['tree_json']['edges']:
        #     scenario.plot_tree_state(edge['from'])
        #     scenario.plot_tree_state(edge['to'])

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
