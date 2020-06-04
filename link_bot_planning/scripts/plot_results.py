#!/usr/bin/env python

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from link_bot_classifiers import classifier_utils
from link_bot_planning.ompl_viz import animate
from link_bot_pycommon.args import my_formatter, int_range_arg
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import model_dirs_from_json
from moonshine.moonshine_utils import numpify, sequence_of_dicts_to_dict_of_sequences, sequence_of_dicts_to_dict_of_np_arrays
from state_space_dynamics import model_utils


def main():
    plt.style.use("paper")
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("results_dir", type=pathlib.Path, help='directory containing metrics.json')
    parser.add_argument("plan_idx", type=int_range_arg, help='which plan to show')
    parser.add_argument("plot_type", choices=['plot', 'animate'], help='how to display')
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--only-collision", action='store_true')
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--headless", action='store_true', help='do not show the window')

    args = parser.parse_args()

    metrics_filename = args.results_dir / "metrics.json"
    data = json.load(metrics_filename.open("r"))
    scenario = get_scenario(data['planner_params']['scenario'])
    classifier_model_dir = pathlib.Path(data['planner_params']['classifier_model_dir'])
    classifier_hparams_filename = classifier_model_dir / 'hparams.json'
    classifier_hparams = json.load(classifier_hparams_filename.open('r'))
    classifier_model = classifier_utils.load_generic_model(classifier_model_dir, scenario=scenario)
    fwd_model_dirs = model_dirs_from_json(data['planner_params']['fwd_model_dir'])
    fwd_model, _ = model_utils.load_generic_model(fwd_model_dirs)

    local_env_h_rows = classifier_hparams['local_env_h_rows']
    local_env_w_cols = classifier_hparams['local_env_w_cols']
    metrics = data['metrics']

    # cc = CollisionCheckerClassifier.check_collision_inflated

    for plan_idx in args.plan_idx:
        # Check if any planned states are in collision
        # environment = metrics['environment']
        # in_collision = False
        # for state in metrics['planned_path']:
        #     # in_collision = in_collision or cc(scenario, local_env_h_rows, local_env_w_cols, environment, ??)
        #
        # if args.only_in_collision and not in_collision:
        #     continue

        plot_plan(args, fwd_model, classifier_model, metrics, plan_idx, scenario)


def plot_plan(args, fwd_model, classifier_model, metrics, plan_idx, scenario):
    metric_for_plan = metrics[plan_idx]
    goal = metric_for_plan['goal']
    labeling_params = classifier_model.dataset_labeling_params
    environment = numpify(metric_for_plan['environment'])
    full_env = np.array(environment['full_env/env'])
    extent = np.array(environment['full_env/extent'])
    planned_path = metric_for_plan['planned_path']
    actual_path = metric_for_plan['actual_path']
    planned_actions = metric_for_plan['actions']
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
            out_filename = args.results_dir / 'plan_{}.png'.format(plan_idx)
            print(f"Saving {out_filename}")
            plt.savefig(out_filename, dpi=200)
        if not args.headless:
            plt.show()
        else:
            plt.close()
    if args.plot_type == 'animate':
        p = sequence_of_dicts_to_dict_of_np_arrays(planned_path)['link_bot']
        a = sequence_of_dicts_to_dict_of_np_arrays(actual_path)['link_bot']
        is_close = np.linalg.norm(p - a, axis=1) < labeling_params['threshold']
        print(is_close)
        anim = animate(environment=environment,
                       scenario=scenario,
                       goal=goal,
                       planned_path=planned_path,
                       actual_path=actual_path,
                       planned_actions=planned_actions,
                       is_close=is_close,
                       fps=args.fps)
        if args.save:
            out_filename = args.results_dir / 'plan_vs_execution_{}.gif'.format(plan_idx)
            print(f"Saving {out_filename}")
            anim.save(out_filename, writer='imagemagick', dpi=50, fps=args.fps)
        if not args.headless:
            plt.show()
        else:
            plt.close()


if __name__ == '__main__':
    main()
