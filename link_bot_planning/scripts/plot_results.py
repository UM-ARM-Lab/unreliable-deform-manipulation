#!/usr/bin/env python

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from link_bot_planning.get_scenario import get_scenario
from link_bot_planning.ompl_viz import plan_vs_execution
from link_bot_planning.params import FullEnvParams
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.link_bot_sdf_utils import compute_extent


def main():
    plt.style.use("paper")
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("plan_idx", type=int, help='which plan to show')
    parser.add_argument("plot_type", choices=['plot', 'animate'], help='how to display')

    args = parser.parse_args()

    metrics_filename = args.results_dir / "metrics.json"
    data = json.load(metrics_filename.open("r"))
    scenario = get_scenario(data['planner_params']['scenario'])
    metrics = data['metrics']
    metric_for_plan = metrics[args.plan_idx]

    goal = metric_for_plan['goal']
    full_env = np.array(metric_for_plan['full_env'])
    h = full_env.shape[0]
    w = full_env.shape[1]
    classifier_hparam_filename = pathlib.Path(data['planner_params']['classifier_model_dir']) / 'hparams.json'
    classifier_hparams = json.load(classifier_hparam_filename.open("r"))
    res = FullEnvParams.from_json(classifier_hparams['classifier_dataset_hparams']['full_env_params']).res
    origin = np.array([h, w]) // 2
    full_env_extent = compute_extent(h, w, res, origin)
    planned_path = metric_for_plan['planned_path']
    actual_path = metric_for_plan['actual_path']

    if args.plot_type == 'plot':
        plt.imshow(np.flipud(full_env), extent=full_env_extent, cmap='Greys')
        ax = plt.gca()
        for state in planned_path:
            scenario.plot_state(ax, state, color='c')
        scenario.plot_goal(ax, goal, label='goal', color='g')
        plt.scatter(planned_path[0, 0], planned_path[0, 1], label='start', color='r')
        plt.legend()
        plt.axis("equal")
        plt.xlim(full_env_extent[0:2])
        plt.ylim(full_env_extent[2:4])
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.show()
    if args.plot_type == 'animate':
        anim = plan_vs_execution(full_env,
                                 full_env_extent,
                                 scenario,
                                 goal,
                                 planned_path,
                                 actual_path)
        out_filename = args.results_dir / 'plan_vs_execution_{}.gif'.format(args.plan_idx)
        anim.save(out_filename, writer='imagemagick', dpi=100)
        # plt.show()


if __name__ == '__main__':
    main()
