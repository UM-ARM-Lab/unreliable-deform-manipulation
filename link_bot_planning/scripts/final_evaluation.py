#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import json
import pathlib
from typing import Optional, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import ompl.util as ou
import tensorflow as tf
from colorama import Fore
from ompl import base as ob

import rospy
from link_bot_data.link_bot_dataset_utils import data_directory
from link_bot_gazebo import gazebo_services
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning import plan_and_execute
from link_bot_planning.get_planner import get_planner
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.ompl_viz import plot_plan
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.params import SimParams
from moonshine.moonshine_utils import listify
from victor import victor_services


class EvalPlannerConfigs(plan_and_execute.PlanAndExecute):

    def __init__(self,
                 planner: MyPlanner,
                 planner_config_name: str,
                 n_plans_per_env: int,
                 n_total_plans: int,
                 verbose: int,
                 planner_params: Dict,
                 sim_params: SimParams,
                 service_provider: GazeboServices,
                 comparison_item_idx: int,
                 seed: int,
                 goal,
                 reset_robot,
                 outdir: pathlib.Path,
                 record: Optional[bool] = False,
                 pause_between_plans: Optional[bool] = False,
                 ):
        super().__init__(planner,
                         n_total_plans=n_total_plans,
                         n_plans_per_env=n_plans_per_env,
                         verbose=verbose,
                         planner_params=planner_params,
                         sim_params=sim_params,
                         service_provider=service_provider,
                         no_execution=False,
                         pause_between_plans=pause_between_plans,
                         seed=seed)
        self.record = record
        self.planner_config_name = planner_config_name
        self.outdir = outdir
        self.seed = seed

        self.metrics = {
            "n_total_plans": n_total_plans,
            "n_targets": n_plans_per_env,
            "planner_params": planner_params,
            "sim_params": sim_params.to_json(),
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
        self.reset_robot = reset_robot

    def on_before_plan(self):
        if self.reset_robot is not None:
            self.service_provider.reset_world(self.verbose, self.reset_robot)

        super().on_before_plan()

    def on_after_plan(self):
        if self.record:
            filename = self.root.absolute() / 'plan-{}.avi'.format(self.total_plan_idx)
            self.service_provider.start_record_trial(str(filename))

        super().on_after_plan()

    def get_goal(self, w_meters, h, environment):
        if self.goal is not None:
            if self.verbose >= 1:
                print("Using Goal {}".format(self.goal))
            return np.array(self.goal)
        else:
            return super().get_goal(w_meters, h, environment)

    def on_execution_complete(self,
                              planned_path: List[Dict],
                              planned_actions: np.ndarray,
                              goal,
                              actual_path: List[Dict],
                              environment: Dict,
                              planner_data: ob.PlannerData,
                              planning_time: float,
                              planner_status: ob.PlannerStatus):
        num_nodes = planner_data.numVertices()

        final_planned_state = planned_path[-1]
        plan_to_goal_error = self.planner.scenario.distance_to_goal(final_planned_state, goal)

        final_state = actual_path[-1]
        execution_to_goal_error = self.planner.scenario.distance_to_goal(final_state, goal)

        plan_to_execution_error = self.planner.scenario.distance(final_state, final_planned_state)

        print("{}: {}".format(self.subfolder, self.successfully_completed_plan_idx))

        planned_path_listified = listify(planned_path)
        planned_actions_listified = listify(planned_actions)
        actual_path_listified = listify(actual_path)

        metrics_for_plan = {
            'planner_status': planner_status.asString(),
            'environment': listify(environment),
            'planned_path': planned_path_listified,
            'actions': planned_actions_listified,
            'actual_path': actual_path_listified,
            'planning_time': planning_time,
            'plan_to_goal_error': plan_to_goal_error,
            'execution_to_goal_error': execution_to_goal_error,
            'plan_to_execution_error': plan_to_execution_error,
            'goal': goal,
            'num_nodes': num_nodes
        }
        self.metrics['metrics'].append(metrics_for_plan)
        metrics_file = self.metrics_filename.open('w')
        json.dump(self.metrics, metrics_file, indent=2)

        plt.figure()
        ax = plt.gca()
        legend = plot_plan(ax=ax,
                           state_space_description=self.planner.state_space_description,
                           scenario=self.planner.scenario,
                           viz_object=self.planner.viz_object,
                           planner_data=planner_data,
                           environment=environment,
                           goal=goal,
                           planned_path=planned_path,
                           planned_actions=None,
                           draw_tree=False,
                           draw_rejected=False)

        self.planner.scenario.plot_state_simple(ax,
                                                final_state,
                                                color='pink',
                                                label='final actual keypoint position',
                                                zorder=5)
        plan_viz_path = self.root / "plan_{}.png".format(self.successfully_completed_plan_idx)
        plt.savefig(plan_viz_path, dpi=600, bbox_extra_artists=(legend,), bbox_inches='tight')

        if self.verbose >= 1:
            print("Final Execution Error: {:0.4f}".format(execution_to_goal_error))
            plt.show()
        else:
            plt.close()

        self.successfully_completed_plan_idx += 1

        if self.record:
            self.service_provider.stop_record_trial()

    def on_complete(self, initial_poses_in_collision):
        self.metrics['initial_poses_in_collision'] = initial_poses_in_collision
        metrics_file = self.metrics_filename.open('w')
        json.dump(self.metrics, metrics_file, indent=2)

    def on_planner_failure(self, start_states, tail_goal_point, environment: Dict, planner_data):
        self.n_failures += 1
        folder = self.failures_root / str(self.n_failures)
        folder.mkdir(parents=True)
        info_file = (folder / 'info.json').open('w')
        info = {
            'start_states': {k: v.tolist() for k, v in start_states.items()},
            'tail_goal_point': tail_goal_point,
            'sdf': {
                'res': environment['full_env/res'],
                'origin': environment['full_env/origin'].tolist(),
                'extent': environment['full_env/extent'].tolist(),
                'data': environment['full_env/env'].tolist(),
            },
        }
        json.dump(info, info_file, indent=2)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
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

    root = pathlib.Path('results') / "{}-compare".format(args.nickname)
    common_output_directory = data_directory(root)
    common_output_directory = pathlib.Path(common_output_directory)
    print(Fore.CYAN + "common output directory: {}".format(common_output_directory) + Fore.RESET)
    if not common_output_directory.is_dir():
        print(Fore.YELLOW + "Creating output directory: {}".format(common_output_directory) + Fore.RESET)
        common_output_directory.mkdir(parents=True)

    rospy.init_node("final_evaluation")
    rospy.set_param('service_provider', args.env_type)

    planners_params = [(json.load(p_params_name.open("r")), p_params_name) for p_params_name in args.planners_params]
    for comparison_idx, (planner_params, p_params_name) in enumerate(planners_params):
        # start at the same seed every time to make the planning environments & plans the same (hopefully?)
        # setting OMPL random seed should have no effect, because I use numpy's random in my sampler?
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)  # not sure if this has any effect

        planner_config_name = p_params_name.stem

        # Start Services
        if args.env_type == 'victor':
            service_provider = victor_services.VictorServices()
        else:
            service_provider = gazebo_services.GazeboServices(planner_params['movable_obstacles'])

        # look up the planner params
        planner, _ = get_planner(planner_params=planner_params,
                                 service_provider=service_provider,
                                 seed=args.seed,
                                 verbose=args.verbose)

        service_provider.setup_env(verbose=args.verbose,
                                   real_time_rate=planner_params['real_time_rate'],
                                   reset_robot=planner_params['reset_robot'],
                                   max_step_size=planner.fwd_model.max_step_size)

        sim_params = SimParams(real_time_rate=planner_params['real_time_rate'],
                               max_step_size=planner.fwd_model.max_step_size,
                               movable_obstacles=[], # unused
                               nudge=False)
        print(Fore.GREEN + "Running {} Trials".format(args.n_total_plans) + Fore.RESET)

        runner = EvalPlannerConfigs(
            planner=planner,
            planner_config_name=planner_config_name,
            n_plans_per_env=args.n_plans_per_env,
            n_total_plans=args.n_total_plans,
            verbose=args.verbose,
            planner_params=planner_params,
            sim_params=sim_params,
            service_provider=service_provider,
            seed=args.seed,
            outdir=common_output_directory,
            comparison_item_idx=comparison_idx,
            reset_robot=planner_params['reset_robot'],
            goal=planner_params['fixed_goal'],
            record=args.record,
            pause_between_plans=args.pause_between_plans
        )
        runner.run()


if __name__ == '__main__':
    main()
