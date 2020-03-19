#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import json
import pathlib
import time
from typing import Optional, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import ompl.util as ou
import rospy
import std_srvs
import tensorflow as tf
from colorama import Fore
from ompl import base as ob

import link_bot_data.link_bot_dataset_utils
from link_bot_gazebo import gazebo_services
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning import plan_and_execute, model_utils
from link_bot_planning.get_scenario import get_scenario
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.ompl_viz import plot_plan
from link_bot_planning.params import SimParams
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter
from victor import victor_services

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


class EvalPlannerConfigs(plan_and_execute.PlanAndExecute):

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
                 goal,
                 reset_gripper_to,
                 outdir: pathlib.Path,
                 record: Optional[bool] = False,
                 pause_between_plans: Optional[bool] = False,
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
            pause_between_plans=pause_between_plans,
            seed=seed)
        self.record = record
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
        self.goal = goal
        self.reset_gripper_to = reset_gripper_to

    def on_before_plan(self):
        if self.reset_gripper_to is not None:
            self.services.reset_world(self.verbose, self.reset_gripper_to)

        super().on_before_plan()

    def on_after_plan(self):
        if self.record:
            filename = self.root.absolute() / 'plan-{}.avi'.format(self.total_plan_idx)
            self.services.start_record_trial(str(filename))

        super().on_after_plan()

    def get_goal(self, w, h, full_env_data):
        if self.goal is not None:
            if self.verbose >= 1:
                print("Using Goal {}".format(self.goal))
            return np.array(self.goal)
        else:
            return super().get_goal(w, h, full_env_data)

    def on_execution_complete(self,
                              planned_path: List[Dict[str, np.ndarray]],
                              planned_actions: np.ndarray,
                              tail_goal_point: np.ndarray,
                              actual_path: List[Dict[str, np.ndarray]],
                              full_env_data: link_bot_sdf_utils.OccupancyData,
                              planner_data: ob.PlannerData,
                              planning_time: float,
                              planner_status: ob.PlannerStatus):
        link_bot_planned_path = planned_path['link_bot']
        link_bot_actual_path = actual_path['link_bot']
        del planned_path, actual_path
        execution_to_goal_error = np.linalg.norm(link_bot_actual_path[-1, 0:2] - tail_goal_point)
        plan_to_goal_error = np.linalg.norm(link_bot_planned_path[-1, 0:2] - tail_goal_point)
        plan_to_execution_error = np.linalg.norm(link_bot_actual_path[-1, 0:2] - link_bot_planned_path[-1, 0:2])
        lengths = [np.linalg.norm(link_bot_planned_path[i] - link_bot_planned_path[i - 1]) for i in
                   range(1, len(link_bot_planned_path))]
        path_length = np.sum(lengths)
        num_nodes = planner_data.numVertices()

        print("{}: {}".format(self.subfolder, self.successfully_completed_plan_idx))

        metrics_for_plan = {
            'planner_status': planner_status.asString(),
            'full_env': full_env_data.data.tolist(),
            'planned_path': link_bot_planned_path.tolist(),
            'actual_path': link_bot_actual_path.tolist(),
            'planning_time': planning_time,
            'final_planning_error': plan_to_goal_error,
            'final_execution_error': execution_to_goal_error,
            'plan_to_execution_error': plan_to_execution_error,
            'path_length': path_length,
            'num_nodes': num_nodes,
            'goal': tail_goal_point,
        }
        self.metrics['metrics'].append(metrics_for_plan)
        metrics_file = self.metrics_filename.open('w')
        json.dump(self.metrics, metrics_file, indent=1)

        plt.figure()
        ax = plt.gca()
        _, legend = plot_plan(ax,
                              self.planner.n_state,
                              self.planner.viz_object,
                              planner_data,
                              full_env_data.data,
                              tail_goal_point,
                              link_bot_planned_path,
                              planned_actions,
                              full_env_data.extent)
        ax.scatter(link_bot_actual_path[-1, 0], link_bot_actual_path[-1, 1], label='final actual tail position', zorder=5)
        plan_viz_path = self.root / "plan_{}.png".format(self.successfully_completed_plan_idx)
        plt.savefig(plan_viz_path, dpi=600, bbox_extra_artists=(legend,), bbox_inches='tight')

        if self.verbose >= 1:
            print("Final Execution Error: {:0.4f}".format(execution_to_goal_error))
            plt.show()
        else:
            plt.close()

        self.successfully_completed_plan_idx += 1

        if self.record:
            self.services.stop_record_trial()

    def on_complete(self, initial_poses_in_collision):
        self.metrics['initial_poses_in_collision'] = initial_poses_in_collision
        metrics_file = self.metrics_filename.open('w')
        json.dump(self.metrics, metrics_file, indent=1)

    def on_planner_failure(self, start_states, tail_goal_point, full_env_data: link_bot_sdf_utils.OccupancyData, planner_data):
        self.n_failures += 1
        folder = self.failures_root / str(self.n_failures)
        folder.mkdir(parents=True)
        image_file = (folder / 'full_env.png')
        info_file = (folder / 'info.json').open('w')
        info = {
            'start_states': dict([(k, v.tolist()) for k, v in start_states.items()]),
            'tail_goal_point': tail_goal_point.tolist(),
            'sdf': {
                'res': full_env_data.resolution.tolist(),
                'origin': full_env_data.origin.tolist(),
                'extent': full_env_data.extent.tolist(),
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
    parser.add_argument("env_type", choices=['victor', 'gazebo'], default='gazebo', help='victor or gazebo')
    parser.add_argument('planners_params', type=pathlib.Path, nargs='+', help='json file(s) describing what should be compared')
    parser.add_argument("--nickname", type=str, help='output will be in results/$nickname-compare-$time', required=True)
    parser.add_argument("--n-total-plans", type=int, default=100, help='total number of plans')
    parser.add_argument("--n-plans-per-env", type=int, default=1, help='number of targets/plans per env')
    parser.add_argument("--pause-between-plans", action='store_true', help='pause between plans')
    parser.add_argument("--seed", '-s', type=int, default=3)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument('--record', action='store_true', help='record')

    args = parser.parse_args()

    print(Fore.CYAN + "Using Seed {}".format(args.seed) + Fore.RESET)

    now = str(int(time.time()))
    root = pathlib.Path('results') / "{}-compare".format(args.nickname)
    common_output_directory = link_bot_data.link_bot_dataset_utils.data_directory(root, now)
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

        scenario = get_scenario(planner_params['scenario'])
        fwd_model, model_path_info = model_utils.load_generic_model(fwd_model_dir, scenario)

        # Start Services
        if args.env_type == 'victor':
            service_provider = victor_services.VictorServices
        else:
            service_provider = gazebo_services.GazeboServices

        services = service_provider.setup_env(verbose=args.verbose,
                                              real_time_rate=planner_params['real_time_rate'],
                                              reset_gripper_to=planner_params['reset_gripper_to'],
                                              max_step_size=fwd_model.max_step_size)

        services.pause(std_srvs.srv.EmptyRequest())

        # look up the planner params
        planner = get_planner_with_model(planner_class_str=planner_type,
                                         fwd_model=fwd_model,
                                         classifier_model_dir=classifier_model_dir,
                                         classifier_model_type=classifier_model_type,
                                         planner_params=planner_params,
                                         services=services,
                                         seed=args.seed)

        sim_params = SimParams(real_time_rate=planner_params['real_time_rate'],
                               max_step_size=planner.fwd_model.max_step_size,
                               goal_padding=0.0,
                               move_obstacles=planner_params['move_obstacles'],
                               nudge=False)
        print(Fore.GREEN + "Running {} Trials".format(args.n_total_plans) + Fore.RESET)

        runner = EvalPlannerConfigs(
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
            comparison_item_idx=comparison_idx,
            reset_gripper_to=planner_params['reset_gripper_to'],
            goal=planner_params['fixed_goal'],
            record=args.record,
            pause_between_plans=args.pause_between_plans
        )
        runner.run()


if __name__ == '__main__':
    main()
