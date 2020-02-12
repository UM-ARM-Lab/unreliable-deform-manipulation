#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import json
import pathlib
import time
from typing import Optional, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import ompl.util as ou
import rospy
import std_srvs
import tensorflow as tf
from colorama import Fore
from ompl import base as ob

from link_bot_data import random_environment_data_utils
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning import my_mpc, model_utils
from link_bot_planning.mpc_planners import get_planner_with_model
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.ompl_viz import plot
from link_bot_planning.params import SimParams
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


class ComputeClassifierMetrics(my_mpc.myMPC):

    def __init__(self,
                 planner: MyPlanner,
                 planner_config_name: str,
                 fwd_model_dir: pathlib.Path,
                 fwd_model_type: str,
                 classifier_model_dir: pathlib.Path,
                 classifier_model_type: str,
                 n_plans_per_env: int,
                 n_total_plans: int,
                 verbose: int,
                 planner_params: Dict,
                 sim_params: SimParams,
                 services: GazeboServices,
                 comparison_item_idx: int,
                 seed: int,
                 outdir: Optional[pathlib.Path] = None,
                 ):
        super().__init__(
            planner,
            n_total_plans=n_total_plans,
            n_plans_per_env=n_plans_per_env,
            verbose=verbose,
            planner_params=planner_params,
            sim_params=sim_params,
            services=services,
            no_execution=False,
            seed=seed)
        self.classifier_model_type = classifier_model_type
        self.planner_config_name = planner_config_name
        self.outdir = outdir
        self.seed = seed

        self.metrics = {
            "fwd_model_dir": str(fwd_model_dir),
            "fwd_model_type": fwd_model_type,
            "classifier_model_dir": str(classifier_model_dir),
            "classifier_model_type": classifier_model_type,
            "n_total_plans": n_total_plans,
            "n_targets": n_plans_per_env,
            "planner_params": planner_params,
            "local_env_params": self.planner.fwd_model.hparams['dynamics_dataset_hparams']['local_env_params'],
            "env_params": sim_params.to_json(),
            "seed": self.seed,
            "metrics": [],
        }
        self.subfolder = "{}_{}".format(self.planner_config_name, comparison_item_idx)
        self.root = self.outdir / self.subfolder
        self.root.mkdir(parents=True)
        print(Fore.CYAN + str(self.root) + Fore.RESET)
        self.metrics_filename = self.root / 'metrics.json'
        self.failures_root = self.root / 'failures'
        self.n_failures = 0
        self.successfully_completed_plan_idx = 0

    def on_execution_complete(self,
                              planned_path: np.ndarray,
                              planned_actions: np.ndarray,
                              tail_goal_point: np.ndarray,
                              planner_local_envs: List[link_bot_sdf_utils.OccupancyData],
                              actual_local_envs: List[link_bot_sdf_utils.OccupancyData],
                              actual_path: np.ndarray,
                              full_env_data: link_bot_sdf_utils.OccupancyData,
                              planner_data: ob.PlannerData,
                              planning_time: float,
                              planner_status: ob.PlannerStatus):
        execution_to_goal_error = np.linalg.norm(actual_path[-1, 0:2] - tail_goal_point)
        plan_to_goal_error = np.linalg.norm(planned_path[-1, 0:2] - tail_goal_point)
        plan_to_execution_error = np.linalg.norm(actual_path[-1, 0:2] - planned_path[-1, 0:2])
        lengths = [np.linalg.norm(planned_path[i] - planned_path[i - 1]) for i in range(1, len(planned_path))]
        path_length = np.sum(lengths)
        num_nodes = planner_data.numVertices()

        print("{}: {}".format(self.subfolder, self.successfully_completed_plan_idx))

        metrics_for_plan = {
            'planner_status': planner_status.asString(),
            'full_env': full_env_data.data.tolist(),
            'planned_path': planned_path.tolist(),
            'actual_path': actual_path.tolist(),
            'planning_time': planning_time,
            'final_planning_error': plan_to_goal_error,
            'final_execution_error': execution_to_goal_error,
            'plan_to_execution_error': plan_to_execution_error,
            'path_length': path_length,
            'num_nodes': num_nodes,
        }
        self.metrics['metrics'].append(metrics_for_plan)
        metrics_file = self.metrics_filename.open('w')
        json.dump(self.metrics, metrics_file, indent=1)

        plt.figure()
        ax = plt.gca()
        legend = plot(ax,
                      self.planner.viz_object,
                      planner_data,
                      full_env_data.data,
                      tail_goal_point,
                      planned_path,
                      planned_actions,
                      full_env_data.extent)
        ax.scatter(actual_path[-1, 0], actual_path[-1, 1], label='final actual tail position', zorder=5)
        plan_viz_path = self.root / "plan_{}.png".format(self.successfully_completed_plan_idx)
        plt.savefig(plan_viz_path, dpi=600, bbox_extra_artists=(legend,), bbox_inches='tight')

        if self.verbose >= 1:
            print("Final Execution Error: {:0.4f}".format(execution_to_goal_error))
            plt.show()
        else:
            plt.close()

        self.successfully_completed_plan_idx += 1

    def on_complete(self, initial_poses_in_collision):
        self.metrics['initial_poses_in_collision'] = initial_poses_in_collision
        metrics_file = self.metrics_filename.open('w')
        json.dump(self.metrics, metrics_file, indent=1)

    def on_planner_failure(self, start, tail_goal_point, full_env_data: link_bot_sdf_utils.OccupancyData):
        self.n_failures += 1
        folder = self.failures_root / str(self.n_failures)
        folder.mkdir(parents=True)
        image_file = (folder / 'full_env.png')
        info_file = (folder / 'info.json').open('w')
        info = {
            'start': start.tolist(),
            'tail_goal_point': tail_goal_point.tolist(),
            'sdf': {
                'res': full_env_data.resolution.tolist(),
                'origin': full_env_data.origin.tolist(),
                'extent': full_env_data.extent,
                'data': full_env_data.data.tolist(),
            },
        }
        json.dump(info, info_file, indent=1)
        plt.imsave(image_file, full_env_data.image > 0)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.logging.set_verbosity(tf.logging.FATAL)
    ou.setLogLevel(ou.LOG_ERROR)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('planners_params', type=pathlib.Path, nargs='+', help='json file(s) describing what should be compared')
    parser.add_argument("--nickname", type=str, help='output will be in results/$nickname-compare-$time',
                        required=True)
    parser.add_argument("--n-total-plans", type=int, default=100, help='total number of plans')
    parser.add_argument("--n-plans-per-env", type=int, default=1, help='number of targets/plans per env')
    parser.add_argument("--seed", '-s', type=int, default=3)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument("--real-time-rate", type=float, default=10.0, help='real time rate')
    parser.add_argument("--goal-threshold", type=float, default=0.1, help='goal radius in meters')
    parser.add_argument('--env-w', type=float, default=5, help='environment width')
    parser.add_argument('--env-h', type=float, default=5, help='environment height')
    parser.add_argument('--max-v', type=float, default=0.15, help='max speed')
    parser.add_argument('--no-move-obstacles', action='store_true', help="don't move obstacles")
    parser.add_argument('--no-nudge', action='store_true', help="don't nudge")
    # TODO: sweep over random epsilon to see how it effects things

    args = parser.parse_args()

    print(Fore.CYAN + "Using Seed {}".format(args.seed) + Fore.RESET)

    now = str(int(time.time()))
    root = pathlib.Path('results') / "{}-compare".format(args.nickname)
    common_output_directory = random_environment_data_utils.data_directory(root, now)
    common_output_directory = pathlib.Path(common_output_directory)
    print(Fore.CYAN + "common output directory: {}".format(common_output_directory) + Fore.RESET)
    if not common_output_directory.is_dir():
        print(Fore.YELLOW + "Creating output directory: {}".format(common_output_directory) + Fore.RESET)
        common_output_directory.mkdir(parents=True)

    rospy.init_node("compare_classifiers")

    initial_object_dict = {
        'moving_box1': [2.0, 0],
        'moving_box2': [-1.5, 0],
        'moving_box3': [-0.5, 1],
        'moving_box4': [1.5, - 2],
        'moving_box5': [-1.5, - 2.0],
        'moving_box6': [-0.5, 2.0],
    }

    planners_params = [(json.load(p_params_name.open("r")), p_params_name) for p_params_name in args.planners_params]
    for comparison_idx, (planner_params, p_params_name) in enumerate(planners_params):
        # start at the same seed every time to make the planning environments & plans the same (hopefully?)
        # setting OMPL random seed should have no effect, because I use numpy's random in my sampler?
        np.random.seed(args.seed)
        tf.random.set_random_seed(args.seed)  # not sure if this has any effect

        planner_config_name = p_params_name.stem
        fwd_model_dir = pathlib.Path(planner_params['fwd_model_dir'])
        fwd_model_type = planner_params['fwd_model_type']
        classifier_model_dir = pathlib.Path(planner_params['classifier_model_dir'])
        classifier_model_type = planner_params['classifier_model_type']
        planner_type = planner_params['planner_type']

        fwd_model, model_path_info = model_utils.load_generic_model(fwd_model_dir, fwd_model_type)

        services = gazebo_utils.setup_gazebo_env(verbose=args.verbose,
                                                 real_time_rate=args.real_time_rate,
                                                 max_step_size=fwd_model.max_step_size,
                                                 reset_world=True,
                                                 initial_object_dict=initial_object_dict)

        services.pause(std_srvs.srv.EmptyRequest())

        # look up the planner params
        planner = get_planner_with_model(planner_class_str=planner_type,
                                         fwd_model=fwd_model,
                                         classifier_model_dir=classifier_model_dir,
                                         classifier_model_type=classifier_model_type,
                                         planner_params=planner_params,
                                         services=services,
                                         seed=args.seed)

        sim_params = SimParams(real_time_rate=args.real_time_rate,
                               max_step_size=planner.fwd_model.max_step_size,
                               goal_padding=0.0,
                               move_obstacles=(not args.no_move_obstacles),
                               nudge=(not args.no_nudge))
        print(Fore.GREEN + "Running {} Trials".format(args.n_total_plans) + Fore.RESET)

        runner = ComputeClassifierMetrics(
            planner=planner,
            planner_config_name=planner_config_name,
            fwd_model_dir=fwd_model_dir,
            fwd_model_type=fwd_model_type,
            classifier_model_dir=classifier_model_dir,
            classifier_model_type=classifier_model_type,
            n_plans_per_env=args.n_plans_per_env,
            n_total_plans=args.n_total_plans,
            verbose=args.verbose,
            planner_params=planner_params,
            sim_params=sim_params,
            services=services,
            seed=args.seed,
            outdir=common_output_directory,
            comparison_item_idx=comparison_idx
        )
        runner.run()


if __name__ == '__main__':
    main()
