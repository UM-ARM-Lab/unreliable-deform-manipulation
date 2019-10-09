#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import matplotlib.pyplot as plt
import rospy
import json
import os
import pathlib
from colorama import Fore
from typing import Optional

import numpy as np
import ompl.util as ou
import tensorflow as tf

from link_bot_data import random_environment_data_utils
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_gazebo.gazebo_utils import get_local_sdf_data
from link_bot_gazebo.srv import LinkBotTrajectoryResponse
from link_bot_planning import shooting_rrt_mpc
from link_bot_planning.shooting_rrt_mpc import PlannerParams, SDFParams, EnvParams
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


class ClasssifierDataCollector(shooting_rrt_mpc.ShootingRRTMPC):

    # TODO: group these arguments more
    def __init__(self,
                 fwd_model_dir: pathlib.Path,
                 fwd_model_type: str,
                 validator_model_dir: pathlib.Path,
                 validator_model_type: str,
                 n_envs: int,
                 n_targets_per_env: int,
                 verbose: int,
                 planner_params: PlannerParams,
                 sdf_params: SDFParams,
                 env_params: EnvParams,
                 n_examples_per_record: int,
                 compression_type: str,
                 outdir: Optional[pathlib.Path] = None):
        super().__init__(fwd_model_dir,
                         fwd_model_type,
                         validator_model_dir,
                         validator_model_type,
                         n_envs,
                         n_targets_per_env,
                         verbose,
                         planner_params,
                         sdf_params,
                         env_params)
        self.n_examples_per_record = n_examples_per_record
        self.compression_type = compression_type
        self.outdir = outdir

        if outdir is not None:
            self.full_output_directory = random_environment_data_utils.data_directory(self.outdir, *self.model_path_info)
            self.full_output_directory = pathlib.Path(self.full_output_directory)
            if not self.full_output_directory.is_dir():
                print(Fore.YELLOW + "Creating output directory: {}".format(self.full_output_directory) + Fore.RESET)
                os.mkdir(self.full_output_directory)
        else:
            self.full_output_directory = None

        with open(pathlib.Path(self.full_output_directory) / 'hparams.json', 'w') as of:
            # FIXME: add all arguments to the constructor here
            options = {
                'dt': self.env_params.dt,
                'n_state': 6,
                'n_action': 2,
            }
            json.dump(options, of, indent=2)

        # This is for saving data
        self.examples = np.ndarray([n_examples_per_record], dtype=object)
        self.example_idx = 0
        self.current_record_traj_idx = 0

        self.planning_times = []

        rospy.init_node('collect_classifier_data')

    def on_plan_complete(self,
                         planned_path: np.ndarray,
                         planned_actions: np.ndarray,
                         planning_time: float):
        self.planning_times.append(planning_time)
        if len(self.planning_times) % 16 == 0:
            print("Planning Time: {:7.3f}s ({:6.3f}s)".format(np.mean(self.planning_times), np.std(self.planning_times)))

    def on_execution_complete(self,
                              planned_path: np.ndarray,
                              planned_actions: np.ndarray,
                              full_sdf_data: link_bot_sdf_utils.SDF,
                              trajectory_execution_result: LinkBotTrajectoryResponse):
        # convert ros message into a T x n_state numpy matrix
        actual_rope_configurations = []
        for configuration in trajectory_execution_result.actual_path:
            np_config = []
            for point in configuration.points:
                np_config.append(point.x)
                np_config.append(point.y)
            actual_rope_configurations.append(np_config)
        actual_rope_configurations = np.array(actual_rope_configurations)

        states = actual_rope_configurations[:-1]
        next_states = actual_rope_configurations[1:]
        planned_states = planned_path[:-1]
        planned_next_states = planned_path[1:]
        for (state, next_state, action, planned_state, planned_next_state) in zip(states, next_states, planned_actions,
                                                                                  planned_states, planned_next_states):
            actual_head_point = state[4:6]
            planner_head_point = planned_state[4:6]
            # compute the local SDF, which may be different for the state in the planner and the state in the real rollout
            actual_local_sdf_data = get_local_sdf_data(self.sdf_params.local_h_rows,
                                                       self.sdf_params.local_w_cols,
                                                       actual_head_point,
                                                       full_sdf_data)
            planner_local_sdf_data = get_local_sdf_data(self.sdf_params.local_h_rows,
                                                        self.sdf_params.local_w_cols,
                                                        planner_head_point,
                                                        full_sdf_data)

            example = ClassifierDataset.make_serialized_example(actual_local_sdf_data.sdf,
                                                                actual_local_sdf_data.extent,
                                                                actual_local_sdf_data.origin,
                                                                planner_local_sdf_data.sdf,
                                                                planner_local_sdf_data.extent,
                                                                planner_local_sdf_data.origin,
                                                                self.sdf_params.local_h_rows,
                                                                self.sdf_params.local_w_cols,
                                                                self.sdf_params.res,
                                                                state,
                                                                next_state,
                                                                action,
                                                                planned_state,
                                                                planned_next_state)

            if self.verbose >= 4:
                plt.figure()
                plt.imshow(planner_local_sdf_data.image > 0, extent=planner_local_sdf_data.extent, zorder=1, alpha=0.5)
                plt.imshow(actual_local_sdf_data.image > 0, extent=actual_local_sdf_data.extent, zorder=1, alpha=0.5)
                plt.scatter(actual_head_point[0], actual_head_point[1], zorder=2)
                plt.scatter(planner_head_point[0], planner_head_point[1], zorder=3)
                plt.axis("equal")
                plt.xlabel("x (m)")
                plt.ylabel("y (m)")
                plt.show()

            self.examples[self.current_record_traj_idx] = example
            self.current_record_traj_idx += 1
            self.example_idx += 1

            if self.current_record_traj_idx == self.n_examples_per_record:
                # save to a TF record
                serialized_dataset = tf.data.Dataset.from_tensor_slices((self.examples))

                end_example_idx = self.example_idx
                start_example_idx = end_example_idx - self.n_examples_per_record
                record_filename = "example_{}_to_{}.tfrecords".format(start_example_idx, end_example_idx - 1)
                full_filename = self.full_output_directory / record_filename
                writer = tf.data.experimental.TFRecordWriter(str(full_filename), compression_type=self.compression_type)
                writer.write(serialized_dataset)
                print("saved {}".format(full_filename))
                self.current_record_traj_idx = 0


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path)
    parser.add_argument("fwd_model_type", choices=['gp', 'llnn', 'rigid'], default='gp')
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--n-envs", type=int, default=32, help='number of environments')
    parser.add_argument("--n-targets-per-env", type=int, default=10, help='number of targets/plans per environment')
    parser.add_argument("--n-examples-per-record", type=int, default=128, help='examples per tfrecord')
    parser.add_argument("--seed", '-s', type=int)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=5.0)
    parser.add_argument("--real-time-rate", type=float, default=1.0, help='real time rate')
    parser.add_argument('--res', '-r', type=float, default=0.03, help='size of cells in meters')
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')
    # Even though the arena is 5m, we need extra padding so that we can request a 1x1 meter local sdf at the corners
    parser.add_argument('--env-w', type=float, default=6, help='environment width')
    parser.add_argument('--env-h', type=float, default=5, help='environment height')
    parser.add_argument('--full-sdf-w', type=float, default=15, help='environment width')
    parser.add_argument('--full-sdf-h', type=float, default=15, help='environment height')
    parser.add_argument('--sdf-cols', type=float, default=100, help='local sdf width')
    parser.add_argument('--sdf-rows', type=float, default=100, help='local sdf width')
    parser.add_argument('--max-v', type=float, default=0.15, help='max speed')

    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
        print("Using seed: ", args.seed)
    np.random.seed(args.seed)

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    planner_params = PlannerParams(timeout=args.planner_timeout, max_v=args.max_v)
    sdf_params = SDFParams(full_h_m=args.env_h,
                           full_w_m=args.env_w,
                           local_h_rows=args.sdf_h,
                           local_w_cols=args.sdf_w,
                           res=args.res)
    env_params = EnvParams(w=args.env_w,
                           h=args.env_h,
                           real_time_rate=args.real_time_rate,
                           goal_padding=0.0)

    data_collector = ClasssifierDataCollector(
        fwd_model_dir=args.fwd_model_dir,
        fwd_model_type=args.fwd_model_type,
        validator_model_dir=pathlib.Path(),
        validator_model_type='none',
        n_envs=args.n_env,
        n_targets_per_env=args.n_targets_per_env,
        verbose=args.verbose,
        planner_params=planner_params,
        sdf_params=sdf_params,
        env_params=env_params,
        n_examples_per_record=args.n_examples_per_records,
        compression_type=args.compression_type,
        outdir=args.outdir
    )
    data_collector.run()


if __name__ == '__main__':
    main()
