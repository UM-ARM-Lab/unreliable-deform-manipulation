#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import rospy
import tensorflow as tf
from colorama import Fore

from link_bot_data.link_bot_dataset_utils import float_feature
from link_bot_gazebo import gazebo_utils
from link_bot_gazebo.gazebo_utils import get_local_occupancy_data
from link_bot_gazebo.msg import LinkBotVelocityAction
from link_bot_planning import model_utils
from link_bot_planning.params import LocalEnvParams
from link_bot_pycommon.args import my_formatter

opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
conf = tf.compat.v1.ConfigProto(gpu_options=opts)
tf.compat.v1.enable_eager_execution(config=conf)

from link_bot_data import random_environment_data_utils
from link_bot_gazebo.srv import WorldControlRequest, LinkBotStateRequest


def generate_trajs(args, fwd_model, full_output_directory, services):
    examples = np.ndarray([args.n_trajs_per_file], dtype=object)

    current_features = {
        'local_env_rows': float_feature(np.array([fwd_model.local_env_params.h_rows])),
        'local_env_cols': float_feature(np.array([fwd_model.local_env_params.w_cols]))
    }

    example_step_idx = 0
    example_idx = 0
    current_example_idx = 0
    for traj_idx in range(args.n_trajs):
        for time_idx in range(args.n_steps_per_traj):
            state_req = LinkBotStateRequest()
            action_msg = LinkBotVelocityAction()

            # Query the current state
            state = services.get_state(state_req)
            head_idx = state.link_names.index("head")
            actual_state = gazebo_utils.points_to_config(state.points)
            head_point = state.points[head_idx]

            # publish the pull command, which will return the target velocity
            vx = np.random.uniform(-0.15, 0.15)
            vy = np.random.uniform(-0.15, 0.15)
            action_msg.gripper1_velocity.x = vx
            action_msg.gripper1_velocity.y = vy
            services.velocity_action_pub.publish(action_msg)

            # let the simulator run
            step = WorldControlRequest()
            step.steps = int(fwd_model.dt / 0.001)  # assuming 0.001s per simulation step
            services.world_control(step)  # this will block until stepping is complete

            # format the tf feature
            head_np = np.array([head_point.x, head_point.y])
            actual_local_env = get_local_occupancy_data(fwd_model.local_env_params.h_rows,
                                                        fwd_model.local_env_params.w_cols,
                                                        fwd_model.local_env_params.res,
                                                        center_point=head_np,
                                                        services=services)

            current_features['{}/state'.format(example_step_idx)] = float_feature(actual_state)
            current_features['{}/action'.format(example_step_idx)] = float_feature(np.array([vx, vy]))
            current_features['{}/actual_local_env/env'.format(example_step_idx)] = float_feature(actual_local_env.data.flatten())
            current_features['{}/actual_local_env/extent'.format(example_step_idx)] = float_feature(
                np.array(actual_local_env.extent))
            current_features['{}/actual_local_env/origin'.format(example_step_idx)] = float_feature(actual_local_env.origin)
            current_features['{}/res'.format(example_step_idx)] = float_feature(np.array([fwd_model.local_env_params.res]))
            current_features['{}/traj_idx'.format(example_step_idx)] = float_feature(np.array([traj_idx]))
            current_features['{}/time_idx '.format(example_step_idx)] = float_feature(np.array([time_idx]))
            current_features['{}/planned_state'.format(example_step_idx)] = float_feature(actual_state)
            current_features['{}/planned_local_env/env'.format(example_step_idx)] = float_feature(
                actual_local_env.data.flatten())
            current_features['{}/planned_local_env/extent'.format(example_step_idx)] = float_feature(
                np.array(actual_local_env.extent))
            current_features['{}/planned_local_env/origin'.format(example_step_idx)] = float_feature(
                np.array(actual_local_env.origin))

            example_step_idx += 1
            if example_step_idx == args.n_steps_per_traj:
                # we have enough time steps for one example, so reset the time step counter and serialize that example
                example_step_idx = 0
                example_proto = tf.train.Example(features=tf.train.Features(feature=current_features))
                example = example_proto.SerializeToString()
                examples[current_example_idx] = example
                current_example_idx += 1
                example_idx += 1
                # reset the current features
                current_features = {
                    'local_env_rows': float_feature(np.array([fwd_model.local_env_params.h_rows])),
                    'local_env_cols': float_feature(np.array([fwd_model.local_env_params.w_cols]))
                }

            if current_example_idx == args.n_trajs_per_file:
                # save to a tf record
                serialized_dataset = tf.data.Dataset.from_tensor_slices((examples))

                end_example_idx = example_idx
                start_example_idx = end_example_idx - args.n_trajs_per_file
                record_filename = "example_{}_to_{}.tfrecords".format(start_example_idx, end_example_idx - 1)
                full_filename = full_output_directory / record_filename
                writer = tf.data.experimental.TFRecordWriter(str(full_filename), compression_type=args.compression_type)
                writer.write(serialized_dataset)
                print("saved {}".format(full_filename))
                current_example_idx = 0


def generate(args):
    rospy.init_node('collect_classifier_data_babbling')

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
        print("Using seed: ", args.seed)
    np.random.seed(args.seed)

    full_output_directory = random_environment_data_utils.data_directory(args.outdir, args.n_trajs)
    if not os.path.isdir(full_output_directory) and args.verbose:
        print(Fore.YELLOW + "Creating output directory: {}".format(full_output_directory) + Fore.RESET)
        os.mkdir(full_output_directory)

    fwd_model, _ = model_utils.load_generic_model(args.fwd_model_dir, args.fwd_model_type)
    with open(pathlib.Path(full_output_directory) / 'hparams.json', 'w') as of:
        options = {
            'dt': fwd_model.dt,
            'seed': args.seed,
            'compression_type': args.compression_type,
            'n_total_plans': 0,
            'n_plans_per_env': 0,
            'verbose': args.verbose,
            'planner_params': None,
            'env_params': None,
            'local_env_params': fwd_model.hparams['dynamics_dataset_hparams']['local_env_params'],
            'steps_per_example': 1,
            'fwd_model_dir': str(args.fwd_model_dir),
            'fwd_model_type': args.fwd_model_type,
            'fwd_model_hparams': fwd_model.hparams,
            'filter_free_space_only': False,
            'labeling': {
                'threshold': 0.2,
                'discard_pre_far': True
            }
        }
        json.dump(options, of, indent=1)

    services = gazebo_utils.setup_gazebo_env(args.verbose, args.real_time_rate, True, None)

    generate_trajs(args, fwd_model, full_output_directory, services)


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("fwd_model_dir", help='', type=pathlib.Path)
    parser.add_argument("fwd_model_type", help='', type=str, choices=['nn', 'llnn', 'gp', 'rigid'])
    parser.add_argument("n_trajs", help='how many trajs to collect', type=int)
    parser.add_argument("--n-steps-per-traj", help='how many steps per traj', type=int, default=10)
    parser.add_argument("outdir")
    parser.add_argument('--env-w', type=float, default=5.0)
    parser.add_argument('--env-h', type=float, default=5.0)
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')
    parser.add_argument("--n-trajs-per-file", type=int, default=256)
    parser.add_argument("--seed", '-s', help='seed', type=int, default=0)
    parser.add_argument("--real-time-rate", help='number of times real time', type=float, default=10)
    parser.add_argument("--verbose", '-v', action="store_true")

    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
