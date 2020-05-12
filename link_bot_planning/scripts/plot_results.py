#!/usr/bin/env python

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from link_bot_pycommon.get_scenario import get_scenario
from link_bot_planning.ompl_viz import plan_vs_execution
from link_bot_pycommon.args import my_formatter


def main():
    plt.style.use("paper")
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("plan_idx", type=int, help='which plan to show')
    parser.add_argument("plot_type", choices=['plot', 'animate'], help='how to display')
    parser.add_argument("--save", action='store_true')

    args = parser.parse_args()

    metrics_filename = args.results_dir / "metrics.json"
    data = json.load(metrics_filename.open("r"))
    scenario = get_scenario(data['planner_params']['scenario'])
    metrics = data['metrics']
    metric_for_plan = metrics[args.plan_idx]

    goal = metric_for_plan['goal']
    environment = metric_for_plan['environment']
    full_env = np.array(environment['full_env/env'])
    extent = np.array(environment['full_env/extent'])
    planned_path = metric_for_plan['planned_path']
    actual_path = metric_for_plan['actual_path']

    if args.plot_type == 'plot':
        plt.figure(figsize=(8, 8))
        plt.imshow(np.flipud(full_env), extent=extent, cmap='Greys')
        ax = plt.gca()
        colormap = cm.winter
        T = len(planned_path)
        for t, state in enumerate(planned_path):
            scenario.plot_state(ax, state, color=colormap(t / T), s=20, zorder=2)
        scenario.plot_goal(ax, goal, label='goal', color='g', zorder=3)
        scenario.plot_state_simple(ax, planned_path[-1], color='m', marker='*', label='end', zorder=3)
        scenario.plot_state_simple(ax, planned_path[0], color='c', label='start', zorder=3)
        plt.legend()
        plt.axis("equal")
        plt.xlim(extent[0:2])
        plt.ylim(extent[2:4])
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        if args.save:
            out_filename = args.results_dir / 'plan_{}.png'.format(args.plan_idx)
            plt.savefig(out_filename, dpi=200)
        plt.show()
    if args.plot_type == 'animate':
        anim = plan_vs_execution(environment=environment,
                                 scenario=scenario,
                                 goal=goal,
                                 planned_path=planned_path,
                                 actual_path=actual_path)
        if args.save:
            out_filename = args.results_dir / 'plan_vs_execution_{}.gif'.format(args.plan_idx)
            anim.save(out_filename, writer='imagemagick', dpi=100)
        plt.show()


if __name__ == '__main__':
    main()
