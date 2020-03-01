#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import pathlib
import time
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import ompl.base as ob
import ompl.util as ou
import rospy
import std_srvs
import tensorflow as tf
from link_bot_planning.shooting_directed_control_sampler import ShootingDirectedControlSampler
from matplotlib.animation import FuncAnimation

from link_bot_data.visualization import plottable_rope_configuration
from link_bot_gazebo import gazebo_services
from link_bot_gazebo.gazebo_services import GazeboServices, get_sdf_data
from link_bot_planning import plan_and_execute
from link_bot_planning.my_planner import MyPlanner, get_planner
from link_bot_planning.ompl_viz import plot
from link_bot_planning.params import SimParams, PlannerParams
from link_bot_pycommon import link_bot_sdf_utils

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


def plot_comparison(outdir, planned_path, actual_rope_configurations, full_sdf_data):
    fig = plt.figure()
    ax = plt.gca()

    actual_handle = plt.plot([], [], label='actual')[0]
    planned_handle = plt.plot([], [], label='planned')[0]
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.imshow(full_sdf_data.image, extent=full_sdf_data.extent)

    def func(t):
        planned_config = planned_path[t]
        actual_config = actual_rope_configurations[t]
        planned_xs, planned_ys = plottable_rope_configuration(planned_config)
        actual_xs, actual_ys = plottable_rope_configuration(actual_config)
        planned_handle.set_xdata(planned_xs)
        planned_handle.set_ydata(planned_ys)
        actual_handle.set_xdata(actual_xs)
        actual_handle.set_ydata(actual_ys)

    outfile = outdir / 'comparison_anim_{}.gif'.format(time.time())
    anim = FuncAnimation(fig, func, interval=250, frames=planned_path.shape[0])
    anim.save(outfile, writer='imagemagick')

    plt.show()


class Executor(plan_and_execute.PlanAndExecute):

    def __init__(self,
                 planner: MyPlanner,
                 verbose: int,
                 planner_params: Dict,
                 sim_params: SimParams,
                 services: GazeboServices,
                 outdir: pathlib.Path):
        super().__init__(
            planner=planner,
            n_total_plans=1,
            n_plans_per_env=1,
            verbose=verbose,
            planner_params=planner_params,
            sim_params=sim_params,
            services=services,
            no_execution=False,
            seed=seed)
        self.outdir = outdir

    def on_plan_complete(self,
                         planned_path: np.ndarray,
                         tail_goal_point: np.ndarray,
                         planned_actions: np.ndarray,
                         full_sdf_data: link_bot_sdf_utils.SDF,
                         planner_data: ob.PlannerData,
                         planning_time: float):
        sampler = ShootingDirectedControlSampler
        plot(sampler, planner_data, full_sdf_data.sdf, tail_goal_point, planned_path, planned_actions, full_sdf_data.extent)
        final_error = np.linalg.norm(planned_path[-1, 0:2] - tail_goal_point)
        lengths = [np.linalg.norm(planned_path[i] - planned_path[i - 1]) for i in range(1, len(planned_path))]
        path_length = np.sum(lengths)
        duration = self.planner.fwd_model.dt * len(planned_path)
        msg = "Final Error: {:0.4f}, Path Length: {:0.4f}, Steps {}, Duration: {:0.2f}s"
        print(msg.format(final_error, path_length, len(planned_path), duration))
        outfile = self.outdir / 'plan_{}.png'.format(time.time())
        plt.savefig(outfile)

    def on_execution_complete(self,
                              planned_path: np.ndarray,
                              planned_actions: np.ndarray,
                              tail_goal_point: np.ndarray,
                              actual_path: Dict[str, np.ndarray],
                              full_env_data: link_bot_sdf_utils.OccupancyData,
                              planner_data: ob.PlannerData,
                              planning_time: float,
                              planner_status: ob.PlannerStatus):
        full_sdf_data = get_sdf_data(env_h=10, env_w=10, res=0.03, services=self.services)
        plot_comparison(self.outdir, planned_path, actual_path['link_bot'], full_sdf_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fwd_model_dir", help="forward model", type=pathlib.Path)
    parser.add_argument("fwd_model_type", choices=['gp', 'llnn', 'rigid'], default='gp')
    parser.add_argument("classifier_model_dir", help="classifier", type=pathlib.Path)
    parser.add_argument("classifier_model_type", choices=['none', 'collision', 'raster'])
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--seed", '-s', type=int, default=10)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=30.0)
    parser.add_argument("--real-time-rate", type=float, default=1.0, help='real time rate')
    parser.add_argument('--planner-env-w', type=float, default=5, help='planner environment width')
    parser.add_argument('--planner-env-h', type=float, default=5, help='planner environment height')
    parser.add_argument('--max-v', type=float, default=0.15, help='max speed')
    parser.add_argument('--max-angle-rad', type=float, default=1, help='maximum deviation from straight rope when sampling')

    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    planner_params = PlannerParams(w=args.planner_env_w,
                                   h=args.planner_env_h,
                                   timeout=args.planner_timeout,
                                   max_v=args.max_v,
                                   goal_threshold=0.1,
                                   max_angle_rad=args.max_angle_rad)
    sim_params = SimParams(real_time_rate=args.real_time_rate,
                           goal_padding=0.0,
                           move_obstacles=False)

    initial_object_dict = {
        'moving_box1': [2.0, 0],
        'moving_box2': [-1.5, 0],
        'moving_box3': [-0.5, 1],
        'moving_box4': [1.5, - 2],
        'moving_box5': [-1.5, - 2.0],
        'moving_box6': [-0.5, 2.0],
    }

    rospy.init_node('planner_with_classifier')

    services = gazebo_services.setup_env(verbose=args.verbose,
                                         real_time_rate=sim_params.real_time_rate,
                                         max_step_size=sim_params.max_step_size,
                                         reset_world=True,
                                         initial_object_dict=initial_object_dict)
    services.pause(std_srvs.srv.EmptyRequest())

    planner, _ = get_planner(planner_class_str='NearestRRT',
                             fwd_model_dir=args.fwd_model_dir,
                             fwd_model_type=args.fwd_model_type,
                             classifier_model_dir=args.classifier_model_dir,
                             classifier_model_type=args.classifier_model_type,
                             planner_params=planner_params,
                             sim_params=sim_params,
                             services=services)

    executer = Executor(
        planner=planner,
        verbose=args.verbose,
        planner_params=planner_params,
        sim_params=sim_params,
        outdir=args.outdir,
        services=services,
    )

    executer.run()


if __name__ == '__main__':
    main()
