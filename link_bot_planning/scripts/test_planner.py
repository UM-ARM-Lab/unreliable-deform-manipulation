#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import json
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import ompl.util as ou
import rospy
import std_srvs
import tensorflow as tf
from ompl import base as ob

from ignition import markers
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning import my_mpc
from link_bot_planning.mpc_planners import get_planner
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.ompl_viz import plot
from link_bot_planning.params import PlannerParams, SimParams
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


class TestWithClassifier(my_mpc.myMPC):

    def __init__(self,
                 planner: MyPlanner,
                 n_targets: int,
                 verbose: int,
                 planner_params: PlannerParams,
                 sim_params: SimParams,
                 services: GazeboServices,
                 no_execution: bool):
        super().__init__(planner=planner,
                         n_total_plans=n_targets,
                         n_plans_per_env=n_targets,
                         verbose=verbose,
                         planner_params=planner_params,
                         sim_params=sim_params,
                         services=services,
                         no_execution=no_execution)

    def on_plan_complete(self,
                         planned_path: np.ndarray,
                         tail_goal_point: np.ndarray,
                         planned_actions: np.ndarray,
                         full_env_data: link_bot_sdf_utils.OccupancyData,
                         planner_data: ob.PlannerData,
                         planning_time: float,
                         planner_status: ob.PlannerStatus):
        final_error = np.linalg.norm(planned_path[-1, 0:2] - tail_goal_point)
        lengths = [np.linalg.norm(planned_path[i] - planned_path[i - 1]) for i in range(1, len(planned_path))]
        path_length = np.sum(lengths)
        duration = self.planner.fwd_model.dt * len(planned_path)

        if self.verbose >= 2:
            planned_final_tail_point_msg = markers.make_marker(id=3, rgb=[0, 0, 1], scale=0.05)
            planned_final_tail_point_msg.pose.position.x = planned_path[-1][0]
            planned_final_tail_point_msg.pose.position.y = planned_path[-1][1]
            markers.publish(planned_final_tail_point_msg)

        msg = "Final Error: {:0.4f}, Path Length: {:0.4f}, Steps {}, Duration: {:0.2f}s"
        print(msg.format(final_error, path_length, len(planned_path), duration))

        plt.figure()
        ax = plt.gca()
        plot(ax, self.planner.viz_object, planner_data, full_env_data.data, tail_goal_point, planned_path, planned_actions,
             full_env_data.extent)
        plt.show()

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
        final_execution_error = np.linalg.norm(actual_path[-1, 0:2] - tail_goal_point)
        print('final execution error {:0.3f}'.format(final_execution_error))


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("params", type=pathlib.Path, help='params json file')
    parser.add_argument("--n-targets", type=int, default=1, help='number of targets/plans')
    parser.add_argument("--seed", '-s', type=int, default=12)
    parser.add_argument("--no-execution", action='store_true', help='do not execute, only plan')
    parser.add_argument('--no-move-obstacles', action='store_true', help="don't move obstacles")
    parser.add_argument('--no-nudge', action='store_true', help="don't nudge")
    parser.add_argument('--reset-world', action='store_true', help="don't nudge")
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=30.0)
    parser.add_argument("--real-time-rate", type=float, default=1.0, help='real time rate')
    parser.add_argument("--max-step-size", type=float, default=0.01, help='seconds per physics step')

    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    planner_params = json.load(args.params.open("r"))
    if args.planner_timeout:
        planner_params['timeout'] = args.planner_timeout

    sim_params = SimParams(real_time_rate=args.real_time_rate,
                           max_step_size=args.max_step_size,
                           goal_padding=0.0,
                           move_obstacles=(not args.no_move_obstacles),
                           nudge=(not args.no_nudge))

    rospy.init_node('test_planner_with_classifier')

    services = gazebo_utils.setup_gazebo_env(verbose=args.verbose,
                                             real_time_rate=sim_params.real_time_rate,
                                             max_step_size=sim_params.max_step_size,
                                             reset_world=args.reset_world,
                                             initial_object_dict=None)
    services.pause(std_srvs.srv.EmptyRequest())

    planner, _ = get_planner(planner_params=planner_params, services=services)

    tester = TestWithClassifier(
        planner=planner,
        n_targets=args.n_targets,
        verbose=args.verbose,
        planner_params=planner_params,
        sim_params=sim_params,
        services=services,
        no_execution=args.no_execution
    )
    tester.run()


if __name__ == '__main__':
    main()
