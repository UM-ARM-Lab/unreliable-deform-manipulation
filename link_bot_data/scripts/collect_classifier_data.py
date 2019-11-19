#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import json
import os
import pathlib
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import ompl.util as ou
import rospy
import std_srvs
import tensorflow as tf
from colorama import Fore
from ompl import base as ob

from link_bot_data import random_environment_data_utils
from link_bot_data.link_bot_dataset_utils import float_feature
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning import my_mpc
from link_bot_planning.mpc_planners import get_planner
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.ompl_viz import plot
from link_bot_planning.params import PlannerParams, LocalEnvParams, EnvParams
from link_bot_planning.shooting_directed_control_sampler import ShootingDirectedControlSampler
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


class ClassifierDataCollector(my_mpc.myMPC):

    def __init__(self,
                 planner: MyPlanner,
                 fwd_model_dir: pathlib.Path,
                 fwd_model_type: str,
                 fwd_model_info: str,
                 n_total_plans: int,
                 n_plans_per_env: int,
                 verbose: int,
                 seed: int,
                 planner_params: PlannerParams,
                 local_env_params: LocalEnvParams,
                 env_params: EnvParams,
                 n_steps_per_example: int,
                 n_examples_per_record: int,
                 compression_type: str,
                 services: GazeboServices,
                 outdir: Optional[pathlib.Path] = None):
        super().__init__(planner,
                         n_total_plans,
                         n_plans_per_env,
                         verbose,
                         planner_params,
                         local_env_params,
                         env_params,
                         services=services,
                         no_execution=False)
        self.fwd_model_dir = fwd_model_dir
        self.fwd_model_type = fwd_model_type
        self.fwd_model_info = fwd_model_info
        self.n_examples_per_record = n_examples_per_record
        self.n_steps_per_example = n_steps_per_example
        self.compression_type = compression_type
        self.outdir = outdir

        if outdir is not None:
            self.full_output_directory = random_environment_data_utils.data_directory(self.outdir, *self.fwd_model_info)
            self.full_output_directory = pathlib.Path(self.full_output_directory)
            if not self.full_output_directory.is_dir():
                print(Fore.YELLOW + "Creating output directory: {}".format(self.full_output_directory) + Fore.RESET)
                os.mkdir(self.full_output_directory)
        else:
            self.full_output_directory = None

        with (self.full_output_directory / 'hparams.json').open('w') as of:
            options = {
                'seed': seed,
                'dt': self.planner.fwd_model.dt,
                'n_state': 6,
                'n_action': 2,
                'compression_type': compression_type,
                'fwd_model_dir': str(self.fwd_model_dir),
                'fwd_model_type': self.fwd_model_type,
                'n_total_plans': n_total_plans,
                'n_plans_per_env': n_plans_per_env,
                'verbose': verbose,
                'planner_params': planner_params.to_json(),
                'env_params': env_params.to_json(),
                'local_env_params': local_env_params.to_json(),
                'sequence_length': self.n_steps_per_example,
                'filter_free_space_only': False,
            }
            json.dump(options, of, indent=2)

        self.planning_times = []

        # NOTE: For datasets collected by planning, the number of steps in the sequence changes, so in order to make the dataset
        #  compatible for training a dynamics function, which assume fixed-length training sequences, we pick a length
        #  and chunk sequence plans. This means a trajectory can contain transitions from sequential plans,
        #  i.e not all from the same plan.
        self.current_features = {}

        # This is for saving data
        self.examples = np.ndarray([n_examples_per_record], dtype=object)
        self.example_idx = 0
        self.example_step_idx = 0
        self.current_example_idx = 0
        self.traj_idx = 0

    def on_plan_complete(self,
                         planned_path: np.ndarray,
                         tail_goal_point: np.ndarray,
                         planned_actions: np.ndarray,
                         full_sdf_data: link_bot_sdf_utils.SDF,
                         planner_data: ob.PlannerData,
                         planning_time: float):
        self.planning_times.append(planning_time)

        if self.verbose >= 2:
            plot(ShootingDirectedControlSampler, planner_data, full_sdf_data.sdf, tail_goal_point, planned_path, planned_actions,
                 full_sdf_data.extent)

        if len(self.planning_times) % 16 == 0:
            print("Planning Time: {:7.3f}s ({:6.3f}s)".format(np.mean(self.planning_times), np.std(self.planning_times)))

    def on_execution_complete(self,
                              planned_path: np.ndarray,
                              planned_actions: np.ndarray,
                              tail_goal_point: np.ndarray,
                              planner_local_envs: List[link_bot_sdf_utils.OccupancyData],
                              actual_local_envs: List[link_bot_sdf_utils.OccupancyData],
                              actual_path: np.ndarray,
                              full_sdf_data: link_bot_sdf_utils.SDF,
                              planner_data: ob.PlannerData,
                              planning_time: float):
        states = actual_path[:-1]
        next_states = actual_path[1:]
        planned_states = planned_path[:-1]
        planned_next_states = planned_path[1:]
        d = zip(states, next_states, planned_actions, planned_states, planned_next_states, actual_local_envs, planner_local_envs)
        for time_idx, data_t in enumerate(d):
            state, next_state, action, planned_state, planned_next_state, actual_local_env, planned_local_env = data_t
            self.current_features['{}/state'.format(self.example_step_idx)] = float_feature(state)
            self.current_features['{}/action'.format(self.example_step_idx)] = float_feature(action)
            self.current_features['{}/actual_local_env/env'.format(self.example_step_idx)] = float_feature(
                actual_local_env.data.flatten())
            self.current_features['{}/actual_local_env/extent'.format(self.example_step_idx)] = float_feature(
                np.array(actual_local_env.extent))
            self.current_features['{}/actual_local_env/origin'.format(self.example_step_idx)] = float_feature(
                actual_local_env.origin)
            self.current_features['{}/res'.format(self.example_step_idx)] = float_feature(np.array([self.local_env_params.res]))
            self.current_features['{}/local_env_rows'.format(self.example_step_idx)] = float_feature(
                np.array([self.local_env_params.h_rows]))
            self.current_features['{}/local_env_cols'.format(self.example_step_idx)] = float_feature(
                np.array([self.local_env_params.w_cols]))
            self.current_features['{}/traj_idx'.format(self.example_step_idx)] = float_feature(np.array([self.traj_idx]))
            self.current_features['{}/time_idx '.format(time_idx)] = float_feature(np.array([self.example_step_idx]))
            self.current_features['{}/planned_state'.format(self.example_step_idx)] = float_feature(planned_state)
            self.current_features['{}/planned_local_env/env'.format(self.example_step_idx)] = float_feature(
                planned_local_env.data.flatten())
            self.current_features['{}/planned_local_env/extent'.format(self.example_step_idx)] = float_feature(
                np.array(planned_local_env.extent))
            self.current_features['{}/planned_local_env/origin'.format(self.example_step_idx)] = float_feature(
                np.array(planned_local_env.origin))

            self.example_step_idx += 1

            if self.verbose >= 4:
                plt.figure()
                plt.imshow(planned_local_env.image > 0, extent=planned_local_env.extent, zorder=1, alpha=0.5)
                plt.imshow(actual_local_env.image > 0, extent=actual_local_env.extent, zorder=1, alpha=0.5)
                plt.axis("equal")
                plt.xlabel("x (m)")
                plt.ylabel("y (m)")
                plt.show()

            time_idx += 1

            if self.example_step_idx == self.n_steps_per_example:
                # we have enough time steps for one example, so reset the time step counter and serialize that example
                self.example_step_idx = 0
                example_proto = tf.train.Example(features=tf.train.Features(feature=self.current_features))
                example = example_proto.SerializeToString()
                self.examples[self.current_example_idx] = example
                self.current_example_idx += 1
                self.example_idx += 1

            if self.current_example_idx == self.n_examples_per_record:
                # save to a tf record
                serialized_dataset = tf.data.Dataset.from_tensor_slices((self.examples))

                end_example_idx = self.example_idx
                start_example_idx = end_example_idx - self.n_examples_per_record
                record_filename = "example_{}_to_{}.tfrecords".format(start_example_idx, end_example_idx - 1)
                full_filename = self.full_output_directory / record_filename
                writer = tf.data.experimental.TFRecordWriter(str(full_filename), compression_type=self.compression_type)
                writer.write(serialized_dataset)
                print("saved {}".format(full_filename))
                self.current_example_idx = 0

        self.traj_idx += 1


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path)
    parser.add_argument("fwd_model_type", choices=['gp', 'llnn', 'rigid'], default='llnn')
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--n-total-plans", type=int, default=32, help='number of environments')
    parser.add_argument("--n-plans-per-env", type=int, default=10, help='number of targets/plans per environment')
    parser.add_argument("--n-steps-per-example", type=int, default=100, help='time steps per example')
    parser.add_argument("--n-examples-per-record", type=int, default=1024, help='examples per tfrecord')
    parser.add_argument("--seed", '-s', type=int)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=10.0)
    parser.add_argument("--real-time-rate", type=float, default=10.0, help='real time rate')
    parser.add_argument('--res', '-r', type=float, default=0.03, help='size of cells in meters')
    parser.add_argument("--compression-type", choices=['', 'zlib', 'gzip'], default='zlib')
    parser.add_argument('--env-w', type=float, default=5, help='environment width')
    parser.add_argument('--env-h', type=float, default=5, help='environment height')
    parser.add_argument('--local-env-cols', type=int, default=50, help='local env width')
    parser.add_argument('--local-env-rows', type=int, default=50, help='local env width')
    parser.add_argument('--max-v', type=float, default=0.15, help='max speed')
    parser.add_argument('--goal-threshold', type=float, default=0.10, help='goal threshold')

    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(0, 100000)
    print("random seed:", args.seed)
    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    planner_params = PlannerParams(timeout=args.planner_timeout, max_v=args.max_v, goal_threshold=args.goal_threshold)
    local_env_params = LocalEnvParams(h_rows=args.local_env_rows,
                                      w_cols=args.local_env_cols,
                                      res=args.res)
    env_params = EnvParams(w=args.env_w,
                           h=args.env_h,
                           real_time_rate=args.real_time_rate,
                           goal_padding=0.0)

    rospy.init_node('collect_classifier_data')

    initial_object_dict = {
        'moving_box1': [2.0, 0],
        'moving_box2': [-1.5, 0],
        'moving_box3': [-0.5, 1],
        'moving_box4': [1.5, - 2],
        'moving_box5': [-1.5, - 2.0],
        'moving_box6': [-0.5, 2.0],
    }

    services = gazebo_utils.setup_gazebo_env(verbose=args.verbose,
                                             real_time_rate=env_params.real_time_rate,
                                             reset_world=True,
                                             initial_object_dict=initial_object_dict)
    services.pause(std_srvs.srv.EmptyRequest())

    planner, fwd_model_info = get_planner(planner_class_str='ShootingRRT',
                                          fwd_model_dir=args.fwd_model_dir,
                                          fwd_model_type=args.fwd_model_type,
                                          classifier_model_dir=pathlib.Path(),
                                          classifier_model_type='none',
                                          planner_params=planner_params,
                                          local_env_params=local_env_params,
                                          env_params=env_params,
                                          services=services)

    data_collector = ClassifierDataCollector(
        planner=planner,
        fwd_model_dir=args.fwd_model_dir,
        fwd_model_type=args.fwd_model_type,
        fwd_model_info=fwd_model_info,
        n_total_plans=args.n_total_plans,
        n_plans_per_env=args.n_plans_per_env,
        verbose=args.verbose,
        seed=args.seed,
        planner_params=planner_params,
        local_env_params=local_env_params,
        env_params=env_params,
        n_steps_per_example=args.n_steps_per_example,
        n_examples_per_record=args.n_examples_per_record,
        compression_type=args.compression_type,
        outdir=args.outdir,
        services=services,
    )

    data_collector.run()


if __name__ == '__main__':
    main()
