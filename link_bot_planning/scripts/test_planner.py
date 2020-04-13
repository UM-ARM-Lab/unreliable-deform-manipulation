#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import json
import pathlib
from typing import Tuple, Optional, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import ompl.util as ou
import rospy
import std_srvs
import tensorflow as tf
from ompl import base as ob

from link_bot_gazebo import gazebo_services
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning import ompl_viz
from link_bot_planning import plan_and_execute
from link_bot_planning.get_planner import get_planner
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.params import SimParams
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter, point_arg
from victor import victor_services

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


class TestWithClassifier(plan_and_execute.PlanAndExecute):

    def __init__(self,
                 planner: MyPlanner,
                 n_targets: int,
                 verbose: int,
                 planner_params: Dict,
                 sim_params: SimParams,
                 services: GazeboServices,
                 no_execution: bool,
                 goal: Optional[Tuple[float, float]],
                 seed: int,
                 draw_tree: Optional[bool] = True,
                 draw_rejected: Optional[bool] = True):
        super().__init__(planner=planner,
                         n_total_plans=n_targets,
                         n_plans_per_env=n_targets,
                         verbose=verbose,
                         planner_params=planner_params,
                         sim_params=sim_params,
                         service_provider=services,
                         no_execution=no_execution,
                         seed=seed,
                         retry_on_failure=False)
        self.goal = goal
        self.draw_tree = draw_tree
        self.draw_rejected = draw_rejected

    def get_goal(self, w_meters, h, full_env_data):
        if self.goal is not None:
            print("Using Goal {}".format(self.goal))
            return np.array(self.goal)
        else:
            return super().get_goal(w_meters, h, full_env_data)

    def on_plan_complete(self,
                         planned_path: List[Dict],
                         goal,
                         planned_actions: np.ndarray,
                         full_env_data: link_bot_sdf_utils.OccupancyData,
                         planner_data: ob.PlannerData,
                         planning_time: float,
                         planner_status: ob.PlannerStatus):
        n_actions = len(planned_actions)
        final_state = planned_path[-1]
        final_error = self.planner.experiment_scenario.distance_to_goal(final_state, goal)

        if self.verbose >= 1:
            self.planner.experiment_scenario.publish_state_marker(self.service_provider.marker_provider, final_state)

        print("Final Error: {:0.4f}, # Actions {}".format(final_error, n_actions))
        print("Planning Time {:0.3f}".format(planning_time))

        if rospy.get_param('service_provider') == 'victor':
            anim = ompl_viz.plan_vs_execution(full_env_data.data,
                                              full_env_data.extent,
                                              self.planner.experiment_scenario,
                                              goal,
                                              planned_path,
                                              actual_path=None)
            plt.show()
        else:
            plt.figure()
            ax = plt.gca()
            legend = ompl_viz.plot_plan(ax,
                                        self.planner.state_space_description,
                                        self.planner.experiment_scenario,
                                        self.planner.viz_object,
                                        planner_data,
                                        full_env_data.data,
                                        goal,
                                        planned_path,
                                        planned_actions,
                                        full_env_data.extent,
                                        draw_tree=self.draw_tree,
                                        draw_rejected=self.draw_rejected)

            plt.savefig("/tmp/.latest-plan.png", dpi=200, bbox_extra_artists=(legend,), bbox_inches='tight')
        plt.show(block=True)

    def on_execution_complete(self,
                              planned_path: List[Dict[str, np.ndarray]],
                              planned_actions: np.ndarray,
                              goal,
                              actual_path: List[Dict[str, np.ndarray]],
                              full_env_data: link_bot_sdf_utils.OccupancyData,
                              planner_data: ob.PlannerData,
                              planning_time: float,
                              planner_status: ob.PlannerStatus):
        print(planned_actions)
        final_planned_state = planned_path[-1]
        plan_to_goal_error = self.planner.experiment_scenario.distance_to_goal(final_planned_state, goal)
        print("Execution to Plan Error: {:.4f}".format(plan_to_goal_error))

        final_state = actual_path[-1]
        execution_to_goal_error = self.planner.experiment_scenario.distance_to_goal(final_state, goal)
        print('Execution to Goal Error: {:0.3f}'.format(execution_to_goal_error))

        anim = ompl_viz.plan_vs_execution(full_env_data.data,
                                          full_env_data.extent,
                                          self.planner.experiment_scenario,
                                          goal,
                                          planned_path,
                                          actual_path)
        anim.save("/tmp/.latest-plan-vs-execution.gif", dpi=200, writer='imagemagick')
        plt.show(block=True)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("service_provider", choices=['victor', 'gazebo'], default='gazebo', help='victor or gazebo')
    parser.add_argument("params", type=pathlib.Path, help='params json file')
    parser.add_argument("--n-targets", type=int, default=1, help='number of targets/plans')
    parser.add_argument("--seed", '-s', type=int, default=5)
    parser.add_argument("--no-execution", action='store_true', help='do not execute, only plan')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument("--planner-timeout", help="time in seconds", type=float)
    parser.add_argument("--real-time-rate", type=float, default=0.0, help='real time rate')
    parser.add_argument("--max-step-size", type=float, default=0.01, help='seconds per physics step')
    parser.add_argument("--goal", type=point_arg, help='x,y in meters')
    parser.add_argument("--debug", action='store_true', help='wait to attach debugger')

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    planner_params = json.load(args.params.open("r"))
    if args.planner_timeout:
        planner_params['timeout'] = args.planner_timeout

    sim_params = SimParams(real_time_rate=args.real_time_rate,
                           max_step_size=planner_params['max_step_size'],
                           movable_obstacles=planner_params['movable_obstacles'],
                           nudge=True)

    rospy.init_node('test_planner_with_classifier')

    if args.debug:
        input("waiting to let you attach debugger...")

    # Start Services
    if args.service_provider == 'victor':
        rospy.set_param('service_provider', 'victor')
        service_provider = victor_services.VictorServices()
    else:
        rospy.set_param('service_provider', 'gazebo')
        service_provider = gazebo_services.GazeboServices()

    service_provider.setup_env(verbose=args.verbose,
                               real_time_rate=sim_params.real_time_rate,
                               reset_robot=planner_params['reset_robot'],
                               max_step_size=sim_params.max_step_size)
    service_provider.pause(std_srvs.srv.EmptyRequest())

    planner, _ = get_planner(planner_params=planner_params, service_provider=service_provider, seed=args.seed,
                             verbose=args.verbose)

    tester = TestWithClassifier(
        planner=planner,
        n_targets=args.n_targets,
        verbose=args.verbose,
        planner_params=planner_params,
        sim_params=sim_params,
        services=service_provider,
        no_execution=args.no_execution,
        goal=args.goal,
        seed=args.seed,
        draw_tree=(args.verbose > 1),
        draw_rejected=(args.verbose > 1)
    )
    tester.run()


if __name__ == '__main__':
    main()
