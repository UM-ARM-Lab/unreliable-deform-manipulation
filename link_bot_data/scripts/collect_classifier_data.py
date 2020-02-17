#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import json
import os
import pathlib
from typing import Optional, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import ompl.util as ou
import rospy
import std_srvs
import tensorflow as tf
from colorama import Fore
from ompl import base as ob

from link_bot_data import random_environment_data_utils
from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning import plan_and_execute, ompl_viz
from link_bot_planning.mpc_planners import get_planner
from link_bot_planning.my_planner import MyPlanner
from link_bot_planning.params import SimParams
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


class ClassifierDataCollector(plan_and_execute.PlanAndExecute):

    def __init__(self,
                 planner: MyPlanner,
                 fwd_model_dir: pathlib.Path,
                 fwd_model_type: str,
                 fwd_model_info: str,
                 n_total_plans: int,
                 n_plans_per_env: int,
                 verbose: int,
                 seed: int,
                 planner_params: Dict,
                 sim_params: SimParams,
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
                         sim_params,
                         services=services,
                         no_execution=False,
                         seed=seed)
        self.hparams_written = False
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

        self.dataset_hparams = {
            'dt': self.planner.fwd_model.dt,
            'seed': seed,
            'compression_type': compression_type,
            'n_total_plans': n_total_plans,
            'n_plans_per_env': n_plans_per_env,
            'verbose': verbose,
            'planner_params': planner_params,
            'sim_params': sim_params.to_json(),
            'local_env_params': self.planner.fwd_model.hparams['dynamics_dataset_hparams']['local_env_params'],
            'full_env_params': self.planner.fwd_model.hparams['dynamics_dataset_hparams']['full_env_params'],
            'sequence_length': self.n_steps_per_example,
            'fwd_model_dir': str(self.fwd_model_dir),
            'fwd_model_type': self.fwd_model_type,
            'fwd_model_hparams': self.planner.fwd_model.hparams,
            'filter_free_space_only': False,
            'n_state': self.planner.fwd_model.hparams['dynamics_dataset_hparams']['n_state'],
            'n_action': self.planner.fwd_model.hparams['dynamics_dataset_hparams']['n_action']
        }

        self.planning_times = []

        # This is for saving data
        self.examples = np.ndarray([n_examples_per_record], dtype=object)
        self.examples_idx = 0
        self.traj_idx = 0

    def on_plan_complete(self,
                         planned_path: Dict[str, np.ndarray],
                         tail_goal_point: np.ndarray,
                         planned_actions: np.ndarray,
                         full_env_data: link_bot_sdf_utils.OccupancyData,
                         planner_data: ob.PlannerData,
                         planning_time: float,
                         planner_status: ob.PlannerStatus):
        self.planning_times.append(planning_time)

        if self.verbose >= 2:
            plt.figure()
            ax = plt.gca()
            ompl_viz.plot(ax,
                          self.planner.viz_object,
                          planner_data,
                          full_env_data.data,
                          tail_goal_point,
                          planned_path['link_bot'],
                          planned_actions,
                          full_env_data.extent)

        if self.verbose >= 1:
            if len(self.planning_times) % 16 == 0:
                print("Planning Time: {:7.3f}s ({:6.3f}s)".format(np.mean(self.planning_times), np.std(self.planning_times)))

    def on_execution_complete(self,
                              planned_path: Dict[str, np.ndarray],
                              planned_actions: np.ndarray,
                              tail_goal_point: np.ndarray,
                              actual_path: Dict[str, np.ndarray],
                              full_env_data: link_bot_sdf_utils.OccupancyData,
                              planner_data: ob.PlannerData,
                              planning_time: float,
                              planner_status: ob.PlannerStatus):

        # write the hparams once we've figured out what objects we're going to have
        if not self.hparams_written:
            self.hparams_written = True
            self.dataset_hparams['actual_state_keys'] = list(actual_path.keys())
            self.dataset_hparams['planned_state_keys'] = list(planned_path.keys())
            with (self.full_output_directory / 'hparams.json').open('w') as of:
                json.dump(self.dataset_hparams, of, indent=2)

        current_features = {
            'full_env/env': float_tensor_to_bytes_feature(full_env_data.data),
            'full_env/extent': float_tensor_to_bytes_feature(full_env_data.extent),
            'full_env/origin': float_tensor_to_bytes_feature(full_env_data.origin),
        }

        for time_idx in range(self.n_steps_per_example):
            # we may have to truncate, or pad the trajectory, depending on the length of the plan
            if time_idx < planned_actions.shape[0]:
                action = planned_actions[time_idx]
            else:
                action = np.zeros(2)

            n_steps = len(next(iter(actual_path.values())))
            for name, object_path in actual_path.items():
                if time_idx < n_steps:
                    state = object_path[time_idx]
                else:
                    state = object_path[-1]
                current_features['{}/state/{}'.format(time_idx, name)] = float_tensor_to_bytes_feature(state)

            for name, object_path in planned_path.items():
                if time_idx < n_steps:
                    state = object_path[time_idx]
                else:
                    state = object_path[-1]
                current_features['{}/planned_state/{}'.format(time_idx, name)] = float_tensor_to_bytes_feature(state)

            current_features['{}/action'.format(time_idx)] = float_tensor_to_bytes_feature(action)
            current_features['{}/res'.format(time_idx)] = float_tensor_to_bytes_feature(self.local_env_params.res)
            current_features['{}/traj_idx'.format(time_idx)] = float_tensor_to_bytes_feature(self.traj_idx)
            current_features['{}/time_idx'.format(time_idx)] = float_tensor_to_bytes_feature(time_idx)

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
    parser.add_argument("n_total_plans", type=int, help='number of plans')
    parser.add_argument("params", type=pathlib.Path, help='params json file')
    parser.add_argument("outdir", type=pathlib.Path)
    parser.add_argument("--n-plans-per-env", type=int, default=8, help='number of targets/plans per environment')
    # if the number of steps in the plan is larger than this number, we truncate.
    # If it is smaller we pad with 0 actions/stationary states
    parser.add_argument("--n-steps-per-example", type=int, default=50, help='time steps per example')
    parser.add_argument("--n-examples-per-record", type=int, default=128, help='examples per tfrecord')
    parser.add_argument("--seed", '-s', type=int)
    parser.add_argument('--verbose', '-v', action='count', default=0, help="use more v's for more verbose, like -vvv")
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=10.0)
    parser.add_argument("--real-time-rate", type=float, default=10.0, help='real time rate')
    parser.add_argument("--max-step-size", type=float, default=0.01, help='seconds per physics step')
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')
    # these define the bounds of the C-space in the planner
    parser.add_argument('--planner-env-w', type=float, default=5.0, help='planner env w')
    parser.add_argument('--planner-env-h', type=float, default=5.0, help='planner env h')
    parser.add_argument('--full-env-w', type=float, default=6.0, help='full env w')
    parser.add_argument('--full-env-h', type=float, default=6.0, help='full env h')
    parser.add_argument('--max-v', type=float, default=0.15, help='max speed')
    parser.add_argument('--no-move-obstacles', action='store_true', help="don't move obstacles")
    parser.add_argument('--no-nudge', action='store_true', help="don't nudge")

    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(0, 100000)
    print("random seed:", args.seed)
    ou.RNG.setSeed(args.seed)
    ou.setLogLevel(ou.LOG_ERROR)

    planner_params = json.load(args.params.open("r"))
    sim_params = SimParams(real_time_rate=args.real_time_rate,
                           max_step_size=args.max_step_size,
                           move_obstacles=(not args.no_move_obstacles),
                           nudge=(not args.no_nudge),
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
                                             real_time_rate=sim_params.real_time_rate,
                                             max_step_size=sim_params.max_step_size,
                                             reset_world=True,
                                             initial_object_dict=initial_object_dict)
    services.pause(std_srvs.srv.EmptyRequest())

    # NOTE: we could make the classifier take a different sized local environment than the dynamics, just a thought.
    planner, fwd_model_info = get_planner(planner_params=planner_params, services=services, seed=args.seed)

    data_collector = ClassifierDataCollector(
        planner=planner,
        fwd_model_dir=planner_params['fwd_model_dir'],
        fwd_model_type=planner_params['fwd_model_type'],
        fwd_model_info=fwd_model_info,
        n_total_plans=args.n_total_plans,
        n_plans_per_env=args.n_plans_per_env,
        verbose=args.verbose,
        seed=args.seed,
        planner_params=planner_params,
        sim_params=sim_params,
        n_steps_per_example=args.n_steps_per_example,
        n_examples_per_record=args.n_examples_per_record,
        compression_type=args.compression_type,
        outdir=args.outdir,
        services=services,
    )

    data_collector.run()


if __name__ == '__main__':
    main()
