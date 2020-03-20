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

from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature, data_directory
from geometry_msgs.msg import Point

from link_bot_planning.get_scenario import get_scenario
from link_bot_planning.params import FullEnvParams, SimParams, CollectDynamicsParams
from link_bot_pycommon import ros_pycommon, link_bot_pycommon
from peter_msgs.srv import ExecuteActionRequest


def sample_delta_pos(action_rng: np.random.RandomState,
                     max_delta_pos: float,
                     gripper_point: Point,
                     goal_env_w: float,
                     goal_env_h: float,
                     last_dx: float,
                     last_dy: float):
    while True:
        if action_rng.uniform(0, 1) < 0.8:
            dx = last_dx
            dy = last_dy
        else:
            delta_pos = action_rng.uniform(0, max_delta_pos)
            direction = action_rng.uniform(-np.pi, np.pi)
            dx = np.cos(direction) * delta_pos
            dy = np.sin(direction) * delta_pos

        half_w = goal_env_w / 2
        half_h = goal_env_h / 2
        if -half_w <= gripper_point.x + dx <= half_w and -half_h <= gripper_point.y + dy <= half_h:
            break

    return dx, dy


def generate_traj(params, args, service_provider, traj_idx, global_t_step, action_rng: np.random.RandomState):
    action_msg = ExecuteActionRequest()

    max_delta_pos = service_provider.get_max_speed() * params.dt

    # At this point, we hope all of the objects have stopped moving, so we can get the environment and assume it never changes
    # over the course of this function
    full_env_data = ros_pycommon.get_occupancy_data(env_w=params.full_env_w_m,
                                                    env_h=params.full_env_h_m,
                                                    res=params.res,
                                                    services=service_provider)

    feature = {
        'full_env/env': float_tensor_to_bytes_feature(full_env_data.data),
        'full_env/extent': float_tensor_to_bytes_feature(full_env_data.extent),
        'full_env/origin': float_tensor_to_bytes_feature(full_env_data.origin),
        'full_env/res': float_tensor_to_bytes_feature(full_env_data.resolution),
    }

    gripper1_dx = gripper1_dy = 0
    for time_idx in range(params.steps_per_traj):
        objects_response = service_provider.get_objects()
        states_dict = {}
        for object in objects_response.objects.objects:
            state = link_bot_pycommon.flatten_named_points(object.points)
            states_dict[object.name] = state

        gripper_point = Point()
        gripper_point.x = states_dict['gripper'][0]
        gripper_point.y = states_dict['gripper'][1]
        gripper1_dx, gripper1_dy = sample_delta_pos(action_rng=action_rng,
                                                    max_delta_pos=max_delta_pos,
                                                    gripper_point=gripper_point,
                                                    goal_env_w=params.goal_w_m,
                                                    goal_env_h=params.goal_h_m,
                                                    last_dx=gripper1_dx,
                                                    last_dy=gripper1_dy)

        action_msg.action.gripper1_delta_pos.x = gripper1_dx
        action_msg.action.gripper1_delta_pos.y = gripper1_dy
        action_msg.action.max_time_per_step = params.dt
        service_provider.execute_action(action_msg)

        feature['{}/action'.format(time_idx)] = float_tensor_to_bytes_feature([gripper1_dx, gripper1_dy])
        for name, state in states_dict.items():
            feature['{}/{}'.format(time_idx, name)] = float_tensor_to_bytes_feature(state)
        feature['{}/traj_idx'.format(time_idx)] = float_tensor_to_bytes_feature(traj_idx)
        feature['{}/time_idx'.format(time_idx)] = float_tensor_to_bytes_feature(time_idx)

        global_t_step += 1

    if args.verbose:
        print(Fore.GREEN + "Trajectory {} Complete".format(traj_idx) + Fore.RESET)

    example_proto = tensorflow.train.Example(features=tensorflow.train.Features(feature=feature))
    example = example_proto.SerializeToString()
    return example, global_t_step


def rearrange_environment(service_provider, params: CollectDynamicsParams, traj_idx, args, env_rng):
    if params.movable_obstacles is not None:
        if len(params.movable_obstacles) > 0 and traj_idx % params.move_objects_every_n == 0:
            service_provider.move_objects(params.max_step_size,
                                          params.movable_obstacles,
                                          params.full_env_w_m,
                                          params.full_env_h_m,
                                          padding=0,
                                          rng=env_rng)


def generate_trajs(service_provider,
                   scenario,
                   params: CollectDynamicsParams,
                   args,
                   full_output_directory,
                   env_rng: np.random.RandomState,
                   action_rng: np.random.RandomState):
    examples = np.ndarray([args.trajs_per_file], dtype=object)
    global_t_step = 0
    for traj_idx in range(args.trajs):
        current_record_traj_idx = traj_idx % args.trajs_per_file

        # Might not do anything, depends on args
        rearrange_environment(service_provider, params, traj_idx, args, env_rng)

        # Generate a new trajectory
        example, global_t_step = generate_traj(params, args, service_provider, traj_idx, global_t_step, action_rng)
        examples[current_record_traj_idx] = example

        # Save the data
        if current_record_traj_idx == args.trajs_per_file - 1:
            # Construct the dataset where each trajectory has been serialized into one big string
            # since TFRecords don't really support hierarchical data structures
            serialized_dataset = tensorflow.data.Dataset.from_tensor_slices((examples))

            end_traj_idx = traj_idx + args.start_idx_offset
            start_traj_idx = end_traj_idx - args.trajs_per_file + 1
            full_filename = os.path.join(full_output_directory, "traj_{}_to_{}.tfrecords".format(start_traj_idx, end_traj_idx))
            writer = tensorflow.data.experimental.TFRecordWriter(full_filename, compression_type='ZLIB')
            writer.write(serialized_dataset)
            print("saved {}".format(full_filename))

        if not args.verbose:
            print(".", end='')
            sys.stdout.flush()


def generate(service_provider, params: CollectDynamicsParams, args):
    rospy.init_node('collect_dynamics_data')

    scenario = get_scenario(args.scenario)

    assert args.trajs % args.trajs_per_file == 0, "num trajs must be multiple of {}".format(args.trajs_per_file)

    full_output_directory = data_directory(args.outdir, args.trajs)
    if not os.path.isdir(full_output_directory) and args.verbose:
        print(Fore.YELLOW + "Creating output directory: {}".format(full_output_directory) + Fore.RESET)
        os.mkdir(full_output_directory)

    full_env_rows = int(params.full_env_h_m / params.res)
    full_env_cols = int(params.full_env_w_m / params.res)
    full_env_params = FullEnvParams(h_rows=full_env_rows, w_cols=full_env_cols, res=params.res)
    sim_params = SimParams(real_time_rate=args.real_time_rate,
                           max_step_size=params.max_step_size,
                           movable_obstacles=params.movable_obstacles)

    states_description = service_provider.get_states_description()
    n_action = service_provider.get_n_action()

    with open(pathlib.Path(full_output_directory) / 'hparams.json', 'w') as of:
        options = {
            'dt': params.dt,
            'max_step_size': params.max_step_size,
            'full_env_params': full_env_params.to_json(),
            'sim_params': sim_params.to_json(),
            'sequence_length': params.steps_per_traj,
            'states_description': states_description,
            'n_action': n_action,
            'scenario': args.scenario,
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
                               max_step_size=params.max_step_size,
                               reset_gripper_to=None)

    generate_trajs(service_provider, scenario, params, args, full_output_directory, env_rng, action_rng)
