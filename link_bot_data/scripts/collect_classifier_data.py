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
from link_bot_planning import my_mpc, ompl_viz
from link_bot_planning.mpc_planners import get_planner
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.params import PlannerParams, EnvParams
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


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
        self.local_env_params = self.planner.fwd_model.local_env_params

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
                'dt': self.planner.fwd_model.dt,
                'seed': seed,
                'compression_type': compression_type,
                'n_total_plans': n_total_plans,
                'n_plans_per_env': n_plans_per_env,
                'verbose': verbose,
                'planner_params': planner_params.to_json(),
                'env_params': env_params.to_json(),
                'local_env_params': self.planner.fwd_model.hparams['dynamics_dataset_hparams']['local_env_params'],
                'sequence_length': self.n_steps_per_example,
                'fwd_model_dir': str(self.fwd_model_dir),
                'fwd_model_type': self.fwd_model_type,
                'fwd_model_hparams': self.planner.fwd_model.hparams,
                'filter_free_space_only': False,
                'labeling': {
                    'threshold': 0.15,
                    'pre_close_threshold': 0.15,
                    'post_close_threshold': 0.21,
                    'discard_pre_far': True
                },
                'n_state': self.planner.fwd_model.hparams['dynamics_dataset_hparams']['n_state'],
                'n_action': self.planner.fwd_model.hparams['dynamics_dataset_hparams']['n_action']
            }
            json.dump(options, of, indent=2)

        self.planning_times = []

        # This is for saving data
        self.examples = np.ndarray([n_examples_per_record], dtype=object)
        self.examples_idx = 0
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
            plt.figure()
            ax = plt.gca()
            ompl_viz.plot(ax,
                          self.planner.viz_object,
                          planner_data,
                          full_sdf_data.sdf,
                          tail_goal_point,
                          planned_path,
                          planned_actions,
                          full_sdf_data.extent)

        if len(self.planning_times) % 16 == 0:
            print("Planning Time: {:7.3f}s ({:6.3f}s)".format(np.mean(self.planning_times), np.std(self.planning_times)))

    def on_execution_complete(self,
                              planned_path: np.ndarray,
                              planned_actions: np.ndarray,
                              tail_goal_point: np.ndarray,
                              planned_local_envs: List[link_bot_sdf_utils.OccupancyData],
                              actual_local_envs: List[link_bot_sdf_utils.OccupancyData],
                              actual_path: np.ndarray,
                              full_sdf_data: link_bot_sdf_utils.SDF,
                              planner_data: ob.PlannerData,
                              planning_time: float):
        current_features = {
            'local_env_rows': float_feature(np.array([self.local_env_params.h_rows])),
            'local_env_cols': float_feature(np.array([self.local_env_params.w_cols]))
        }

        for time_idx in range(self.n_steps_per_example):
            # we may have to truncate, or pad the trajectory, depending on the length of the plan
            if time_idx < planned_actions.shape[0]:
                action = planned_actions[time_idx]
            else:
                action = np.zeros(2)

            if time_idx < planned_path.shape[0]:
                planned_state = planned_path[time_idx]
                planned_local_env = planned_local_envs[time_idx]
            else:
                planned_state = planned_path[-1]
                planned_local_env = planned_local_envs[-1]

            if time_idx < actual_path.shape[0]:
                state = actual_path[time_idx]
                actual_local_env = actual_local_envs[time_idx]
            else:
                state = actual_path[-1]
                actual_local_env = actual_local_envs[-1]

            current_features['{}/state'.format(time_idx)] = float_feature(state)
            current_features['{}/action'.format(time_idx)] = float_feature(action)
            current_features['{}/actual_local_env/env'.format(time_idx)] = float_feature(actual_local_env.data.flatten())
            current_features['{}/actual_local_env/extent'.format(time_idx)] = float_feature(np.array(actual_local_env.extent))
            current_features['{}/actual_local_env/origin'.format(time_idx)] = float_feature(actual_local_env.origin)
            current_features['{}/res'.format(time_idx)] = float_feature(np.array([self.local_env_params.res]))
            current_features['{}/traj_idx'.format(time_idx)] = float_feature(np.array([self.traj_idx]))
            current_features['{}/time_idx '.format(time_idx)] = float_feature(np.array([time_idx]))
            current_features['{}/planned_state'.format(time_idx)] = float_feature(planned_state)
            current_features['{}/planned_local_env/env'.format(time_idx)] = float_feature(planned_local_env.data.flatten())
            current_features['{}/planned_local_env/extent'.format(time_idx)] = float_feature(np.array(planned_local_env.extent))
            current_features['{}/planned_local_env/origin'.format(time_idx)] = float_feature(np.array(planned_local_env.origin))

        self.traj_idx += 1

        example_proto = tf.train.Example(features=tf.train.Features(feature=current_features))
        example = example_proto.SerializeToString()
        self.examples[self.examples_idx] = example
        self.examples_idx += 1

        if self.examples_idx == self.n_examples_per_record:
            # save to a tf record
            serialized_dataset = tf.data.Dataset.from_tensor_slices((self.examples))

            end_example_idx = self.traj_idx
            start_example_idx = end_example_idx - self.n_examples_per_record
            record_filename = "example_{}_to_{}.tfrecords".format(start_example_idx, end_example_idx - 1)
            full_filename = self.full_output_directory / record_filename
            writer = tf.data.experimental.TFRecordWriter(str(full_filename), compression_type=self.compression_type)
            writer.write(serialized_dataset)
            print("saved {}".format(full_filename))
            self.examples_idx = 0


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help="load this saved forward model file", type=pathlib.Path)
    parser.add_argument("fwd_model_type", choices=['gp', 'llnn', 'nn', 'rigid'], default='llnn')
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--classifier-model-dir", help="load this saved forward model file", type=pathlib.Path)
    parser.add_argument("--classifier-model-type", choices=['collision', 'none', 'raster'], default='none')
    parser.add_argument("--n-total-plans", type=int, default=1024, help='number of environments')
    parser.add_argument("--n-plans-per-env", type=int, default=1, help='number of targets/plans per environment')
    # if the number of steps in the plan is larger than this number, we truncate.
    # If it is smaller we pad with 0 actions/stationary states
    parser.add_argument("--n-steps-per-example", type=int, default=50, help='time steps per example')
    parser.add_argument("--n-examples-per-record", type=int, default=8, help='examples per tfrecord')
    parser.add_argument("--seed", '-s', type=int)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=10.0)
    parser.add_argument("--real-time-rate", type=float, default=10.0, help='real time rate')
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')
    parser.add_argument('--env-w', type=float, default=5, help='environment width')
    parser.add_argument('--env-h', type=float, default=5, help='environment height')
    parser.add_argument('--max-v', type=float, default=0.15, help='max speed')
    parser.add_argument('--goal-threshold', type=float, default=0.10, help='goal threshold')
    parser.add_argument('--no-move-obstacles', action='store_true', help="don't move obstacles")

    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(0, 100000)
    print("random seed:", args.seed)
    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    planner_params = PlannerParams(timeout=args.planner_timeout,
                                   max_v=args.max_v,
                                   goal_threshold=args.goal_threshold,
                                   random_epsilon=0.05)
    env_params = EnvParams(w=args.env_w,
                           h=args.env_h,
                           real_time_rate=args.real_time_rate,
                           move_obstacles=(not args.no_move_obstacles),
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

    # NOTE: we could make the classifier take a different sized local environment than the dynamics, just a thought.
    planner, fwd_model_info = get_planner(planner_class_str='ShootingRRT',
                                          fwd_model_dir=args.fwd_model_dir,
                                          fwd_model_type=args.fwd_model_type,
                                          classifier_model_dir=args.classifier_model_dir,
                                          classifier_model_type=args.classifier_model_type,
                                          planner_params=planner_params,
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
