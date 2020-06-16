#!/usr/bin/env python
import json
import os
import pathlib
import sys
from time import perf_counter
from typing import Dict

import numpy as np
import tensorflow as tf
from colorama import Fore

import rospy
from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature, data_directory, \
    dict_of_float_tensors_to_bytes_feature
from link_bot_pycommon import ros_pycommon
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.link_bot_sdf_utils import extent_to_env_shape
from link_bot_pycommon.params import CollectDynamicsParams, Environment
from link_bot_pycommon.ros_pycommon import get_states_dict


# TODO: make this a class, to reduce number of arguments passed
def generate_traj(scenario: ExperimentScenario,
                  params: CollectDynamicsParams,
                  service_provider,
                  traj_idx: int,
                  global_t_step: int,
                  action_rng: np.random.RandomState,
                  verbose: int,
                  action_description: Dict,
                  states_description: Dict):
    if params.no_obstacles:
        rows, cols, channels = extent_to_env_shape(params.extent, params.res)
        origin = np.array([rows // 2, cols // 2, channels // 2], dtype=np.int32)
        env = np.zeros([rows, cols, channels], dtype=np.float32)
        environment = Environment(env=env, res=params.res, origin=origin, extent=params.extent).to_dict()
    else:
        # At this point, we hope all of the objects have stopped moving, so we can get the environment and assume it never changes
        # over the course of this function
        environment = ros_pycommon.get_environment_for_extents_3d(extent=params.extent,
                                                                  res=params.res,
                                                                  service_provider=service_provider,
                                                                  robot_name=scenario.robot_name())

    feature = dict_of_float_tensors_to_bytes_feature(environment)
    feature['traj_idx'] = float_tensor_to_bytes_feature(traj_idx)

    delta_position = None
    actions = {'delta_position': []}
    states = {k: [] for k in states_description.keys()}
    time_indices = []
    for time_idx in range(params.steps_per_traj):
        state_dict = get_states_dict(service_provider)
        delta_position = scenario.sample_action(environment=environment,
                                                service_provider=service_provider,
                                                state=state_dict,
                                                last_action=delta_position,
                                                params=params,
                                                action_rng=action_rng)

        current_position = scenario.state_to_gripper_position(state_dict)
        actions['delta_position'].append(delta_position)
        for state_name, state in state_dict.items():
            states[state_name].append(state)
        time_indices.append(time_idx)

        absolute_action = current_position + delta_position
        scenario.execute_action({'position': absolute_action, 'timeout': [params.dt]})

        global_t_step += 1

    feature.update(dict_of_float_tensors_to_bytes_feature(states))
    feature.update(dict_of_float_tensors_to_bytes_feature(actions))
    feature['time_idx'] = float_tensor_to_bytes_feature(time_indices)

    if verbose:
        print(Fore.GREEN + "Trajectory {} Complete".format(traj_idx) + Fore.RESET)

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    example = example_proto.SerializeToString()
    return example, global_t_step


def rearrange_environment(service_provider, params: CollectDynamicsParams, env_rng):
    if params.movable_obstacles is not None:
        if len(params.movable_obstacles) > 0:
            movable_obstacles = params.movable_obstacles
            service_provider.move_objects_randomly(env_rng, movable_obstacles)


def generate_trajs(service_provider,
                   scenario: ExperimentScenario,
                   params: CollectDynamicsParams,
                   args,
                   full_output_directory,
                   env_rng: np.random.RandomState,
                   action_rng: np.random.RandomState,
                   action_description: Dict,
                   states_description: Dict):
    examples = np.ndarray([params.trajs_per_file], dtype=object)
    global_t_step = 0
    last_record_t = perf_counter()
    for traj_idx in range(args.trajs):
        # Might not do anything, depends on args
        rearrange_environment(service_provider, params, env_rng)

        # Generate a new trajectory
        example, global_t_step = generate_traj(scenario=scenario,
                                               params=params,
                                               service_provider=service_provider,
                                               traj_idx=traj_idx,
                                               global_t_step=global_t_step,
                                               action_rng=action_rng,
                                               verbose=args.verbose,
                                               action_description=action_description,
                                               states_description=states_description)
        current_record_traj_idx = traj_idx % params.trajs_per_file
        examples[current_record_traj_idx] = example

        # Save the data
        if current_record_traj_idx == params.trajs_per_file - 1:
            # Construct the dataset where each trajectory has been serialized into one big string
            # since TFRecords don't really support hierarchical data structures
            serialized_dataset = tf.data.Dataset.from_tensor_slices((examples))

            end_traj_idx = traj_idx + args.start_idx_offset
            start_traj_idx = end_traj_idx - params.trajs_per_file + 1
            full_filename = os.path.join(full_output_directory,
                                         "traj_{}_to_{}.tfrecords".format(start_traj_idx, end_traj_idx))
            writer = tf.data.experimental.TFRecordWriter(full_filename, compression_type='ZLIB')
            writer.write(serialized_dataset)
            now = perf_counter()
            dt_record = now - last_record_t
            print("saved {} ({:5.1f}s)".format(full_filename, dt_record))
            last_record_t = now

        if not args.verbose:
            print(".", end='')
            sys.stdout.flush()


def generate(service_provider, params: CollectDynamicsParams, args):
    rospy.init_node('collect_dynamics_data')
    scenario = get_scenario(args.scenario)

    assert args.trajs % params.trajs_per_file == 0, "num trajs must be multiple of {}".format(params.trajs_per_file)

    full_output_directory = data_directory(args.outdir, args.trajs)
    if not os.path.isdir(full_output_directory) and args.verbose:
        print(Fore.YELLOW + "Creating output directory: {}".format(full_output_directory) + Fore.RESET)
        os.mkdir(full_output_directory)

    states_description = service_provider.get_states_description()
    action_description = service_provider.get_action_description()

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
    print(Fore.CYAN + "Using seed: {}".format(args.seed) + Fore.RESET)

    with open(pathlib.Path(full_output_directory) / 'hparams.json', 'w') as of:
        options = {
            'seed': args.seed,
            'n_trajs': args.trajs,
            'data_collection_params': params.to_json(),
            'states_description': states_description,
            'action_description': action_description,
            'scenario': args.scenario,
        }
        json.dump(options, of, indent=2)

    np.random.seed(args.seed)
    env_rng = np.random.RandomState(args.seed)
    action_rng = np.random.RandomState(args.seed)

    service_provider.setup_env(verbose=args.verbose,
                               real_time_rate=args.real_time_rate,
                               max_step_size=params.max_step_size)

    generate_trajs(service_provider=service_provider,
                   scenario=scenario,
                   params=params,
                   args=args,
                   full_output_directory=full_output_directory,
                   env_rng=env_rng,
                   action_rng=action_rng,
                   action_description=action_description,
                   states_description=states_description)
