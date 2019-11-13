#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import json
import pathlib
import time
from typing import Optional, List

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
from link_bot_planning import my_mpc
from link_bot_planning.mpc_planners import get_planner
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.ompl_viz import plot
from link_bot_planning.params import PlannerParams, LocalEnvParams, EnvParams
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


class ComputeClassifierMetrics(my_mpc.myMPC):

    def __init__(self,
                 planner: MyPlanner,
                 fwd_model_dir: pathlib.Path,
                 fwd_model_type: str,
                 classifier_model_dir: pathlib.Path,
                 classifier_model_type: str,
                 n_plans_per_env: int,
                 n_total_plans: int,
                 verbose: int,
                 planner_params: PlannerParams,
                 local_env_params: LocalEnvParams,
                 env_params: EnvParams,
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
            local_env_params=local_env_params,
            env_params=env_params,
            services=services,
            no_execution=False)
        self.classifier_model_type = classifier_model_type
        self.outdir = outdir
        self.seed = seed

        self.metrics = {
            "fwd_model_dir": str(fwd_model_dir),
            "fwd_model_type": fwd_model_type,
            "classifier_model_dir": str(classifier_model_dir),
            "classifier_model_type": classifier_model_type,
            "n_total_plans": n_total_plans,
            "n_targets": n_plans_per_env,
            "planner_params": planner_params.to_json(),
            "local_env_params": local_env_params.to_json(),
            "env_params": env_params.to_json(),
            "seed": self.seed,
            "metrics": [],
        }
        subfolder = "{}_{}".format(self.classifier_model_type, comparison_item_idx)
        self.root = self.outdir / subfolder
        self.root.mkdir(parents=True)
        print(Fore.CYAN + str(self.root) + Fore.RESET)
        self.metrics_filename = self.root / 'metrics.json'
        self.successfully_completed_plan_idx = 0

    def on_execution_complete(self,
                              planned_path: np.ndarray,
                              planned_actions: np.ndarray,
                              tail_goal_point: np.ndarray,
                              planner_local_envs: List[link_bot_sdf_utils.OccupancyData],
                              actual_local_envs: List[link_bot_sdf_utils.OccupancyData],
                              actual_path: np.ndarray,
                              full_sdf_data: link_bot_sdf_utils.SDF,
                              planner_data: ob.PlannerData,
                              planning_time: float):
        final_execution_error = np.linalg.norm(actual_path[-1, 0:2] - tail_goal_point)
        final_planning_error = np.linalg.norm(planned_path[-1, 0:2] - tail_goal_point)
        lengths = [np.linalg.norm(planned_path[i] - planned_path[i - 1]) for i in range(1, len(planned_path))]
        path_length = np.sum(lengths)

        print("{}: {}".format(self.classifier_model_type, self.successfully_completed_plan_idx))

        metrics_for_plan = {
            'planning_time': planning_time,
            'final_planning_error': final_planning_error,
            'final_execution_error': final_execution_error,
            'path_length': path_length,
        }
        self.metrics['metrics'].append(metrics_for_plan)
        metrics_file = self.metrics_filename.open('w')
        json.dump(self.metrics, metrics_file, indent=1)

        full_binary = full_sdf_data.sdf > 0
        plt.figure()
        ax = plt.gca()
        plot(ax, self.planner.viz_object, planner_data, full_binary, tail_goal_point, planned_path, planned_actions,
             full_sdf_data.extent)
        ax.scatter(actual_path[-1, 0], actual_path[-1, 1], label='final actual tail position')
        plan_viz_path = self.root / "plan_{}.png".format(self.successfully_completed_plan_idx)
        plt.savefig(plan_viz_path, dpi=600)

        if self.verbose >= 1:
            print("Final Execution Error: {:0.4f}".format(final_execution_error))
            plt.show()
        else:
            plt.close()

        self.successfully_completed_plan_idx += 1

    def on_complete(self, initial_poses_in_collision):
        self.metrics['initial_poses_in_collision'] = initial_poses_in_collision
        metrics_file = self.metrics_filename.open('w')
        json.dump(self.metrics, metrics_file, indent=1)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="forward model", type=pathlib.Path)
    parser.add_argument("fwd_model_type", choices=['gp', 'llnn', 'rigid'], default='gp')
    parser.add_argument('comparison', type=pathlib.Path, help='json file describing what should be compared')
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--n-total-plans", type=int, default=10, help='total number of plans')
    parser.add_argument("--n-plans-per-env", type=int, default=5, help='number of targets/plans per env')
    parser.add_argument("--seed", '-s', type=int, default=3)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=15.0)
    parser.add_argument("--real-time-rate", type=float, default=10.0, help='real time rate')
    parser.add_argument('--res', '-r', type=float, default=0.03, help='size of cells in meters')
    parser.add_argument('--env-w', type=float, default=5, help='environment width')
    parser.add_argument('--env-h', type=float, default=5, help='environment height')
    parser.add_argument('--max-v', type=float, default=0.15, help='max speed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    now = str(int(time.time()))
    common_output_directory = random_environment_data_utils.data_directory(args.outdir, now)
    common_output_directory = pathlib.Path(common_output_directory)
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

    services = gazebo_utils.setup_gazebo_env(verbose=args.verbose,
                                             real_time_rate=args.real_time_rate,
                                             reset_world=True,
                                             initial_object_dict=initial_object_dict)
    services.pause(std_srvs.srv.EmptyRequest())

    comparisons = json.load(args.comparison.open("r"))
    for comparison_idx, item_of_comparison in enumerate(comparisons):
        classifier_model_dir = pathlib.Path(item_of_comparison['classifier_model_dir'])
        classifier_model_type = item_of_comparison['classifier_model_type']

        model_hparams_file = classifier_model_dir / 'hparams.json'
        if model_hparams_file.exists():
            model_hparams = json.load(model_hparams_file.open('r'))
            local_env_rows, local_env_cols = model_hparams['local_env_shape']
        else:
            local_env_shape = [50, 50]
            local_env_rows, local_env_cols = local_env_shape
            print(Fore.YELLOW + "no model hparams, assuming local env is {}".format(local_env_shape) + Fore.RESET)

        planner_params = PlannerParams(timeout=args.planner_timeout, max_v=args.max_v, goal_threshold=0.1)
        local_env_params = LocalEnvParams(h_rows=local_env_rows,
                                          w_cols=local_env_cols,
                                          res=args.res)
        env_params = EnvParams(w=args.env_w,
                               h=args.env_h,
                               real_time_rate=args.real_time_rate,
                               goal_padding=0.0)

        planner = get_planner(planner_class_str='ShootingRRT',
                              fwd_model_dir=args.fwd_model_dir,
                              fwd_model_type=args.fwd_model_type,
                              classifier_model_dir=classifier_model_dir,
                              classifier_model_type=classifier_model_type,
                              planner_params=planner_params,
                              local_env_params=local_env_params,
                              env_params=env_params,
                              services=services)

        runner = ComputeClassifierMetrics(
            planner=planner,
            fwd_model_dir=args.fwd_model_dir,
            fwd_model_type=args.fwd_model_type,
            classifier_model_dir=classifier_model_dir,
            classifier_model_type=classifier_model_type,
            n_plans_per_env=args.n_plans_per_env,
            n_total_plans=args.n_total_plans,
            verbose=args.verbose,
            planner_params=planner_params,
            local_env_params=local_env_params,
            env_params=env_params,
            services=services,
            seed=args.seed,
            outdir=common_output_directory,
            comparison_item_idx=comparison_idx
        )
        runner.run()


if __name__ == '__main__':
    main()
