#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import os
import pathlib
from typing import Optional, List

import numpy as np
import ompl.util as ou
import rospy
import tensorflow as tf
from colorama import Fore
from ompl import base as ob

from link_bot_data import random_environment_data_utils
from link_bot_planning import shooting_rrt_mpc
from link_bot_planning.ompl_viz import plot
from link_bot_planning.params import PlannerParams, SDFParams, EnvParams
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


class TestWithClassifier(shooting_rrt_mpc.ShootingRRTMPC):

    # TODO: group these arguments more
    def __init__(self,
                 fwd_model_dir: pathlib.Path,
                 fwd_model_type: str,
                 validator_model_dir: pathlib.Path,
                 validator_model_type: str,
                 n_targets: int,
                 verbose: int,
                 planner_params: PlannerParams,
                 sdf_params: SDFParams,
                 env_params: EnvParams,
                 outdir: Optional[pathlib.Path] = None):
        # TODO: add keywords here
        super().__init__(fwd_model_dir,
                         fwd_model_type,
                         validator_model_dir,
                         validator_model_type,
                         1,
                         n_targets,
                         verbose,
                         planner_params,
                         sdf_params,
                         env_params)
        self.outdir = outdir

        if outdir is not None:
            self.full_output_directory = random_environment_data_utils.data_directory(self.outdir, *self.model_path_info)
            self.full_output_directory = pathlib.Path(self.full_output_directory)
            if not self.full_output_directory.is_dir():
                print(Fore.YELLOW + "Creating output directory: {}".format(self.full_output_directory) + Fore.RESET)
                os.mkdir(self.full_output_directory)
        else:
            self.full_output_directory = None

    def on_plan_complete(self,
                         planned_path: np.ndarray,
                         tail_goal_point: np.ndarray,
                         planned_actions: np.ndarray,
                         full_sdf_data: link_bot_sdf_utils.SDF,
                         planning_time: float):
        # TODO: make planner_data an argument to this function
        planner_data = ob.PlannerData(self.rrt.si)
        self.rrt.planner.getPlannerData(planner_data)
        plot(planner_data, full_sdf_data.sdf, tail_goal_point, planned_path, planned_actions, self.env_params.extent)
        final_error = np.linalg.norm(planned_path[-1, 0:2] - tail_goal_point)
        lengths = [np.linalg.norm(planned_path[i] - planned_path[i - 1]) for i in range(1, len(planned_path))]
        path_length = np.sum(lengths)
        duration = self.fwd_model.dt * len(planned_path)
        msg = "Final Error: {:0.4f}, Path Length: {:0.4f}, Steps {}, Duration: {:0.2f}s"
        print(msg.format(final_error, path_length, len(planned_path), duration))

    def on_execution_complete(self,
                              planned_path: np.ndarray,
                              planned_actions: np.ndarray,
                              planner_local_sdfs: List[link_bot_sdf_utils.SDF],
                              actual_local_sdfs: List[link_bot_sdf_utils.SDF],
                              actual_path: np.ndarray):
        pass


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="forward model", type=pathlib.Path)
    parser.add_argument("fwd_model_type", choices=['gp', 'llnn', 'rigid'], default='gp')
    parser.add_argument("validator_model_dir", help="validator", type=pathlib.Path)
    parser.add_argument("validator_model_type", choices=['none', 'raster'], default='raster')
    parser.add_argument("--outdir", type=pathlib.Path)
    parser.add_argument("--n-targets", type=int, default=1, help='number of targets/plans')
    parser.add_argument("--seed", '-s', type=int, default=3)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=15.0)
    parser.add_argument("--real-time-rate", type=float, default=1.0, help='real time rate')
    parser.add_argument('--res', '-r', type=float, default=0.03, help='size of cells in meters')
    parser.add_argument('--env-w', type=float, default=5, help='environment width')
    parser.add_argument('--env-h', type=float, default=5, help='environment height')
    parser.add_argument('--full-sdf-w', type=float, default=15, help='environment width')
    parser.add_argument('--full-sdf-h', type=float, default=15, help='environment height')
    parser.add_argument('--sdf-cols', type=float, default=100, help='local sdf width')
    parser.add_argument('--sdf-rows', type=float, default=100, help='local sdf width')
    parser.add_argument('--max-v', type=float, default=0.15, help='max speed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    rospy.init_node('planner_with_classifier')

    # TODO: pull out setup and pass in services

    planner_params = PlannerParams(timeout=args.planner_timeout, max_v=args.max_v)
    sdf_params = SDFParams(full_h_m=args.env_h,
                           full_w_m=args.env_w,
                           local_h_rows=args.sdf_rows,
                           local_w_cols=args.sdf_cols,
                           res=args.res)
    env_params = EnvParams(w=args.env_w,
                           h=args.env_h,
                           real_time_rate=args.real_time_rate,
                           goal_padding=0.0)

    tester = TestWithClassifier(
        fwd_model_dir=args.fwd_model_dir,
        fwd_model_type=args.fwd_model_type,
        validator_model_dir=args.validator_model_dir,
        validator_model_type=args.validator_model_type,
        n_targets=args.n_targets,
        verbose=args.verbose,
        planner_params=planner_params,
        sdf_params=sdf_params,
        env_params=env_params,
        outdir=args.outdir
    )
    tester.run()


if __name__ == '__main__':
    main()
