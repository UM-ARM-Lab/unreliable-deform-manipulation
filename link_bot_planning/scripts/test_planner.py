#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import json
import pathlib
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
from link_bot_planning import ompl_viz
from link_bot_planning.params import SimParams
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.link_bot_pycommon import point_arg

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


class TestWithClassifier(my_mpc.myMPC):

    def __init__(self,
                 planner: MyPlanner,
                 n_targets: int,
                 verbose: int,
                 planner_params: Dict,
                 sim_params: SimParams,
                 services: GazeboServices,
                 no_execution: bool,
                 goal: Optional[Tuple[float, float]],
                 seed: int):
        super().__init__(planner=planner,
                         n_total_plans=n_targets,
                         n_plans_per_env=n_targets,
                         verbose=verbose,
                         planner_params=planner_params,
                         sim_params=sim_params,
                         services=services,
                         no_execution=no_execution,
                         seed=seed)
        self.goal = goal

    def get_goal(self, w, h, head_point, env_padding, full_env_data):
        if self.goal is not None:
            print("Using Goal {}".format(self.goal))
            return np.array(self.goal)
        else:
            return super().get_goal(w, h, head_point, env_padding, full_env_data)

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

        num_nodes = planner_data.numVertices()
        print("num nodes {}".format(num_nodes))
        print("planning time {:0.4f}".format(planning_time))

        plt.figure()
        ax = plt.gca()
        legend = ompl_viz.plot(ax, self.planner.viz_object, planner_data, full_env_data.data, tail_goal_point, planned_path,
                               planned_actions, full_env_data.extent)
        plt.savefig("/tmp/.latest-plan.png", dpi=600, bbox_extra_artists=(legend,), bbox_inches='tight')
        plt.show(block=True)

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
        print('Execution to Goal Error: {:0.3f}'.format(execution_to_goal_error))

        print("Execution to Plan Error:")
        for t in range(planned_path.shape[0] - 1):
            planned_s = planned_path[t]
            actual_s = actual_path[t]
            distance = np.linalg.norm(planned_s - actual_s)
            speed = np.linalg.norm(planned_actions[t])
            print("t={:3d}, error={:6.3f}m, speed={:6.3}m/s".format(t, distance, speed))

        anim = ompl_viz.plan_vs_execution(full_env_data.data, tail_goal_point, planned_path, actual_path, full_env_data.extent)
        anim.save("/tmp/.latest-plan-vs-execution.gif", dpi=300, writer='imagemagick')
        plt.show(block=True)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("params", type=pathlib.Path, help='params json file')
    parser.add_argument("--n-targets", type=int, default=1, help='number of targets/plans')
    parser.add_argument("--seed", '-s', type=int, default=5)
    parser.add_argument("--no-execution", action='store_true', help='do not execute, only plan')
    parser.add_argument('--no-move-obstacles', action='store_true', help="don't move obstacles")
    parser.add_argument('--no-nudge', action='store_true', help="don't nudge")
    parser.add_argument('--reset-world', action='store_true', help="reset world")
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=30.0)
    parser.add_argument("--real-time-rate", type=float, default=1.0, help='real time rate')
    parser.add_argument("--max-step-size", type=float, default=0.01, help='seconds per physics step')
    parser.add_argument("--goal", type=point_arg, help='x,y in meters')

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)
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

    planner, _ = get_planner(planner_params=planner_params, services=services, seed=args.seed)

    tester = TestWithClassifier(
        planner=planner,
        n_targets=args.n_targets,
        verbose=args.verbose,
        planner_params=planner_params,
        sim_params=sim_params,
        services=services,
        no_execution=args.no_execution,
        goal=args.goal,
        seed=args.seed,
    )
    tester.run()


if __name__ == '__main__':
    main()