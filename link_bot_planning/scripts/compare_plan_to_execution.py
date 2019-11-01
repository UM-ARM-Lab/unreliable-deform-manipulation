#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import pathlib
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import ompl.base as ob
import ompl.util as ou
import rospy
import std_srvs
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from link_bot_data.visualization import plottable_rope_configuration
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import GazeboServices, get_sdf_data
from link_bot_planning import my_mpc
from link_bot_planning.ompl_viz import plot
from link_bot_planning.params import EnvParams, LocalEnvParams, PlannerParams
from link_bot_planning.shooting_directed_control_sampler import ShootingDirectedControlSampler
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


class Executor(my_mpc.myMPC):

    def __init__(self,
                 fwd_model_dir: pathlib.Path,
                 fwd_model_type: str,
                 verbose: int,
                 planner_params: PlannerParams,
                 local_env_params: LocalEnvParams,
                 env_params: EnvParams,
                 services: GazeboServices,
                 outdir: pathlib.Path):
        super().__init__(fwd_model_dir=fwd_model_dir,
                         fwd_model_type=fwd_model_type,
                         classifier_model_dir=pathlib.Path(),
                         classifier_model_type='none',
                         n_envs=1,
                         n_targets_per_env=1,
                         verbose=verbose,
                         planner_params=planner_params,
                         local_env_params=local_env_params,
                         env_params=env_params,
                         services=services,
                         no_execution=False)
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
        duration = self.fwd_model.dt * len(planned_path)
        msg = "Final Error: {:0.4f}, Path Length: {:0.4f}, Steps {}, Duration: {:0.2f}s"
        print(msg.format(final_error, path_length, len(planned_path), duration))
        outfile = self.outdir / 'plan_{}.png'.format(time.time())
        plt.savefig(outfile)

    def on_execution_complete(self,
                              planned_path: np.ndarray,
                              planned_actions: np.ndarray,
                              planner_local_sdfs: List[link_bot_sdf_utils.SDF],
                              actual_local_sdfs: List[link_bot_sdf_utils.SDF],
                              actual_path: np.ndarray):
        full_sdf_data = get_sdf_data(env_h=10, env_w=10, res=0.03, services=self.services)
        plot_comparison(self.outdir, planned_path, actual_path, full_sdf_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fwd_model_dir", help="forward model", type=pathlib.Path)
    parser.add_argument("fwd_model_type", choices=['gp', 'llnn', 'rigid'], default='gp')
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--seed", '-s', type=int, default=3)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=30.0)
    parser.add_argument("--real-time-rate", type=float, default=1.0, help='real time rate')
    parser.add_argument('--res', '-r', type=float, default=0.03, help='size of cells in meters')
    parser.add_argument('--env-w', type=float, default=5, help='environment width')
    parser.add_argument('--env-h', type=float, default=5, help='environment height')
    parser.add_argument('--local-env-cols', type=float, default=100, help='local env cols')
    parser.add_argument('--local-env-rows', type=float, default=100, help='local env rows')
    parser.add_argument('--max-v', type=float, default=0.15, help='max speed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    planner_params = PlannerParams(timeout=args.planner_timeout, max_v=args.max_v)
    local_env_params = LocalEnvParams(h_rows=args.local_env_rows,
                                      w_cols=args.local_env_cols,
                                      res=args.res)
    env_params = EnvParams(w=args.env_w,
                           h=args.env_h,
                           real_time_rate=args.real_time_rate,
                           goal_padding=0.0)

    initial_object_dict = {
        'moving_box1': [2.0, 0],
        'moving_box2': [-1.5, 0],
        'moving_box3': [-0.5, 1],
        'moving_box4': [1.5, - 2],
        'moving_box5': [-1.5, - 2.0],
        'moving_box6': [-0.5, 2.0],
    }

    rospy.init_node('planner_with_classifier')

    services = gazebo_utils.setup_gazebo_env(verbose=args.verbose,
                                             real_time_rate=env_params.real_time_rate,
                                             reset_world=True,
                                             initial_object_dict=initial_object_dict)
    services.pause(std_srvs.srv.EmptyRequest())

    executer = Executor(
        fwd_model_dir=args.fwd_model_dir,
        fwd_model_type=args.fwd_model_type,
        verbose=args.verbose,
        planner_params=planner_params,
        local_env_params=local_env_params,
        env_params=env_params,
        outdir=args.outdir,
        services=services,
    )

    executer.run()


if __name__ == '__main__':
    main()
