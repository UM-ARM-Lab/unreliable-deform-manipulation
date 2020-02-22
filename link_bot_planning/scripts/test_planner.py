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
from link_bot_planning import plan_and_execute
from link_bot_planning.mpc_planners import get_planner
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning import ompl_viz
from link_bot_planning.params import SimParams
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter, point_arg

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
                 seed: int):
        super().__init__(planner=planner,
                         n_total_plans=n_targets,
                         n_plans_per_env=n_targets,
                         verbose=verbose,
                         planner_params=planner_params,
                         sim_params=sim_params,
                         services=services,
                         no_execution=no_execution,
                         seed=seed,
                         retry_on_failure=False)
        self.goal = goal

    def get_goal(self, w, h, head_point, env_padding, full_env_data):
        if self.goal is not None:
            print("Using Goal {}".format(self.goal))
            return np.array(self.goal)
        else:
            return super().get_goal(w, h, head_point, env_padding, full_env_data)

    def on_planner_failure(self,
                           start: np.ndarray,
                           tail_goal_point: np.ndarray,
                           full_env_data: link_bot_sdf_utils.OccupancyData,
                           planner_data: ob.PlannerData):
        plt.figure()
        ax = plt.gca()
        legend = ompl_viz.plot(ax,
                               self.planner.n_state,
                               self.planner.viz_object,
                               planner_data,
                               full_env_data.data,
                               tail_goal_point,
                               None,
                               None,
                               full_env_data.extent)
        plt.show(block=True)

    def on_plan_complete(self,
                         planned_path: np.ndarray,
                         tail_goal_point: np.ndarray,
                         planned_actions: np.ndarray,
                         full_env_data: link_bot_sdf_utils.OccupancyData,
                         planner_data: ob.PlannerData,
                         planning_time: float,
                         planner_status: ob.PlannerStatus):
        link_bot_planned_path = planned_path['link_bot']
        final_error = np.linalg.norm(link_bot_planned_path[-1, 0:2] - tail_goal_point)
        lengths = [np.linalg.norm(link_bot_planned_path[i] - link_bot_planned_path[i - 1]) for i in
                   range(1, len(link_bot_planned_path))]
        path_length = np.sum(lengths)
        duration = self.planner.fwd_model.dt * len(link_bot_planned_path)

        if self.verbose >= 2:
            planned_final_tail_point_msg = markers.make_marker(id=3, rgb=[0, 0, 1], scale=0.05)
            planned_final_tail_point_msg.pose.position.x = link_bot_planned_path[-1][0]
            planned_final_tail_point_msg.pose.position.y = link_bot_planned_path[-1][1]
            markers.publish(planned_final_tail_point_msg)

        msg = "Final Error: {:0.4f}, Path Length: {:0.4f}, Steps {}, Duration: {:0.2f}s"
        print(msg.format(final_error, path_length, len(link_bot_planned_path), duration))

        num_nodes = planner_data.numVertices()
        print("num nodes {}".format(num_nodes))
        print("planning time {:0.4f}".format(planning_time))

        plt.figure()
        ax = plt.gca()
        plot_data_dict, legend = ompl_viz.plot(ax,
                               self.planner.n_state,
                               self.planner.viz_object,
                               planner_data,
                               full_env_data.data,
                               tail_goal_point,
                               link_bot_planned_path,
                               planned_actions,
                               full_env_data.extent)

        np.savez("/tmp/.latest-plan.npz", **plot_data_dict)
        plt.savefig("/tmp/.latest-plan.png", dpi=600, bbox_extra_artists=(legend,), bbox_inches='tight')
        plt.show(block=True)

    def on_execution_complete(self,
                              planned_path: np.ndarray,
                              planned_actions: np.ndarray,
                              tail_goal_point: np.ndarray,
                              actual_path: Dict[str, np.ndarray],
                              full_env_data: link_bot_sdf_utils.OccupancyData,
                              planner_data: ob.PlannerData,
                              planning_time: float,
                              planner_status: ob.PlannerStatus):
        execution_to_goal_error = np.linalg.norm(actual_path['link_bot'][-1, 0:2] - tail_goal_point)
        print('Execution to Goal Error: {:0.3f}'.format(execution_to_goal_error))

        # Convert from the actual space to the planning space, which may be identity, or may be some reduction
        link_bot_actual_path = actual_path['link_bot']
        link_bot_planned_path = planned_path['link_bot']
        print("Execution to Plan Error: {:.4f}".format(np.linalg.norm(link_bot_planned_path[-1] - link_bot_actual_path[-1])))

        anim = ompl_viz.plan_vs_execution(full_env_data.data,
                                          tail_goal_point,
                                          link_bot_planned_path,
                                          link_bot_actual_path,
                                          full_env_data.extent)
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
    parser.add_argument('--reset-world', action='store_true', help="reset world")
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument("--planner-timeout", help="time in seconds", type=float)
    parser.add_argument("--real-time-rate", type=float, default=0.0, help='real time rate')
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
                           nudge=False)

    rospy.init_node('test_planner_with_classifier')

    services = gazebo_utils.setup_env(verbose=args.verbose,
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
