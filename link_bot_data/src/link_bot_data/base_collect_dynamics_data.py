#!/usr/bin/env python
from __future__ import print_function, division

import json
import os
import pathlib
import sys

import numpy as np
import rospy
import tensorflow
from colorama import Fore

import ignition.markers
from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature, data_directory
from link_bot_planning.params import LocalEnvParams, FullEnvParams, SimParams
from link_bot_pycommon import ros_pycommon, link_bot_pycommon
from peter_msgs.srv import ExecuteActionRequest


def sample_delta_pos(action_rng, max_delta_pos, last_dx, last_dy):
    # TODO: consider resetting on out-of-bounds? right now there is no notion of bounds
    if action_rng.uniform(0, 1) < 0.8:
        gripper1_dx = last_dx
        gripper1_dy = last_dy
    else:
        delta_pos = action_rng.uniform(0, max_delta_pos)
        direction = action_rng.uniform(-np.pi, np.pi)
        gripper1_dx = np.cos(direction) * delta_pos
        gripper1_dy = np.sin(direction) * delta_pos

    return gripper1_dx, gripper1_dy


def generate_traj(args, service_provider, traj_idx, global_t_step, action_rng: np.random.RandomState):
    action_msg = ExecuteActionRequest()

    max_delta_pos = ros_pycommon.get_max_speed() * args.dt

    # At this point, we hope all of the objects have stopped moving, so we can get the environment and assume it never changes
    # over the course of this function
    full_env_data = ros_pycommon.get_occupancy_data(env_w=args.env_w, env_h=args.env_h, res=args.res, services=service_provider)

    feature = {
        'local_env_rows': float_tensor_to_bytes_feature([args.local_env_rows]),
        'local_env_cols': float_tensor_to_bytes_feature([args.local_env_cols]),
        'full_env/env': float_tensor_to_bytes_feature(full_env_data.data),
        'full_env/extent': float_tensor_to_bytes_feature(full_env_data.extent),
        'full_env/origin': float_tensor_to_bytes_feature(full_env_data.origin),
        'full_env/res': float_tensor_to_bytes_feature(full_env_data.resolution),
        'local_env/res': float_tensor_to_bytes_feature(args.resolution)
    }

    gripper1_dx = gripper1_dy = 0
    for time_idx in range(args.steps_per_traj):
        objects_response = service_provider.get_objects()
        states_dict = {}
        # FIXME: how to get state in a generic way?
        for object in objects_response.objects:
            state = link_bot_pycommon.flatten_named_points(object.points)
            local_env_point = ?
            local_env_data = ros_pycommon.get_local_occupancy_data(args.local_env_rows,
                                                                   args.local_env_cols,
                                                                   args.res,
                                                                   center_point=local_env_point,
                                                                   services=service_provider)
            states_dict[object.name] = state

        gripper1_dx, gripper1_dy = sample_delta_pos(action_rng, max_delta_pos, gripper1_dx, gripper1_dy)

        action_msg.action.gripper1_delta_pos.x = gripper1_dx
        action_msg.action.gripper1_delta_pos.y = gripper1_dy
        action_msg.action.max_time_per_step = args.dt
        service_provider.execute_action(action_msg)

        feature['{}/action'.format(time_idx)] = float_tensor_to_bytes_feature([gripper1_dx, gripper1_dy])
        for name, state in states_dict.items():
            feature['{}/state/{}'.format(time_idx, name)] = float_tensor_to_bytes_feature(state)
        feature['{}/traj_idx'.format(time_idx)] = float_tensor_to_bytes_feature(traj_idx)
        feature['{}/time_idx'.format(time_idx)] = float_tensor_to_bytes_feature(time_idx)

        global_t_step += 1

    if args.verbose:
        print(Fore.GREEN + "Trajectory {} Complete".format(traj_idx) + Fore.RESET)

    example_proto = tensorflow.train.Example(features=tensorflow.train.Features(feature=feature))
    example = example_proto.SerializeToString()
    return example, global_t_step


def generate_trajs(service_provider, args, full_output_directory, env_rng: np.random.RandomState,
                   action_rng: np.random.RandomState):
    examples = np.ndarray([args.trajs_per_file], dtype=object)
    global_t_step = 0
    for i in range(args.trajs):
        current_record_traj_idx = i % args.trajs_per_file

        if not args.no_obstacles and i % args.move_objects_every_n == 0:
            objects = ['moving_box{}'.format(i) for i in range(1, 7)]
            service_provider.move_objects(args.max_step_size,
                                          objects,
                                          args.env_w,
                                          args.env_h,
                                          padding=0.5,
                                          rng=env_rng)

        # Generate a new trajectory
        example, global_t_step = generate_traj(args, service_provider, i, global_t_step, action_rng)
        examples[current_record_traj_idx] = example

        # Save the data
        if current_record_traj_idx == args.trajs_per_file - 1:
            # Construct the dataset where each trajectory has been serialized into one big string
            # since TFRecords don't really support hierarchical data structures
            serialized_dataset = tensorflow.data.Dataset.from_tensor_slices((examples))

            end_traj_idx = i + args.start_idx_offset
            start_traj_idx = end_traj_idx - args.trajs_per_file + 1
            full_filename = os.path.join(full_output_directory, "traj_{}_to_{}.tfrecords".format(start_traj_idx, end_traj_idx))
            writer = tensorflow.data.experimental.TFRecordWriter(full_filename, compression_type='ZLIB')
            writer.write(serialized_dataset)
            print("saved {}".format(full_filename))

        if not args.verbose:
            print(".", end='')
            sys.stdout.flush()


def generate(service_provider, args):
    rospy.init_node('collect_dynamics_data')

    assert args.trajs % args.trajs_per_file == 0, "num trajs must be multiple of {}".format(args.trajs_per_file)

    full_output_directory = data_directory(args.outdir, args.trajs)
    if not os.path.isdir(full_output_directory) and args.verbose:
        print(Fore.YELLOW + "Creating output directory: {}".format(full_output_directory) + Fore.RESET)
        os.mkdir(full_output_directory)

    local_env_params = LocalEnvParams(h_rows=args.local_env_rows, w_cols=args.local_env_cols, res=args.res)
    full_env_cols = int(args.env_w / args.res)
    full_env_rows = int(args.env_h / args.res)
    full_env_params = FullEnvParams(h_rows=full_env_rows, w_cols=full_env_cols, res=args.res)
    sim_params = SimParams(real_time_rate=args.real_time_rate,
                           max_step_size=args.max_step_size,
                           goal_padding=0.5,
                           move_obstacles=not args.no_obstacles)

    states_description = service_provider.get_states_description()

    with open(pathlib.Path(full_output_directory) / 'hparams.json', 'w') as of:
        options = {
            'dt': args.dt,
            'max_step_size': args.max_step_size,
            'local_env_params': local_env_params.to_json(),
            'full_env_params': full_env_params.to_json(),
            'sim_params': sim_params.to_json(),
            'sequence_length': args.steps_per_traj,
            'states_description': states_description,
        }
        json.dump(options, of, indent=1)

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
    print(Fore.CYAN + "Using seed: {}".format(args.seed) + Fore.RESET)
    np.random.seed(args.seed)
    env_rng = np.random.RandomState(args.seed)
    action_rng = np.random.RandomState(args.seed)

    service_provider.setup_env(verbose=args.verbose,
                               real_time_rate=args.real_time_rate,
                               max_step_size=args.max_step_size,
                               reset_gripper_to=None,
                               initial_object_dict=None)

    generate_trajs(service_provider, args, full_output_directory, env_rng, action_rng)
