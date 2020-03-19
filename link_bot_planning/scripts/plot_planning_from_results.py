#!/usr/bin/env python

import matplotlib.pyplot as plt
import argparse
import numpy as np
import json
import pathlib

from link_bot_planning.get_scenario import get_scenario
from link_bot_planning.params import LocalEnvParams
from link_bot_pycommon.args import my_formatter, point_arg
from link_bot_pycommon.link_bot_sdf_utils import compute_extent


def main():
    plt.style.use("paper")
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("scenario", choices=['tether', 'link_bot'], help='directory containing metrics.json')
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("plan_idx", type=int, help='which plan to show')
    parser.add_argument("--goal", type=point_arg, help='goal for drawing')

    args = parser.parse_args()

    scenario = get_scenario(args.scenario)

    metrics_filename = args.results_dir / "metrics.json"
    data = json.load(metrics_filename.open("r"))
    metrics = data['metrics']
    metric_for_plan = metrics[args.plan_idx]

    full_env = np.array(metric_for_plan['full_env'])
    h = full_env.shape[0]
    w = full_env.shape[1]
    res = LocalEnvParams.from_json(data['local_env_params']).res
    origin = np.array([h, w]) // 2
    full_env_extent = compute_extent(h, w, res, origin)

    plt.imshow(np.flipud(full_env), extent=full_env_extent, cmap='Greys')
    ax = plt.gca()
    planned_path = np.array(metric_for_plan['planned_path'])
    for state in planned_path:
        state_dict = {
            'link_bot': state,
        }
        scenario.plot_state(ax, state_dict, color='c')
    scenario.plot_goal(ax, args.goal, label='goal', color='g')
    plt.scatter(planned_path[0, 0], planned_path[0, 1], label='start', color='r')
    plt.legend()
    plt.axis("equal")
    plt.xlim([-2.5,2.5])
    plt.ylim([-2.5,2.5])
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()


if __name__ == '__main__':
    main()
