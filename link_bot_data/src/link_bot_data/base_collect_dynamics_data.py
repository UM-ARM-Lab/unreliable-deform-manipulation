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
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.link_bot_sdf_utils import extent_to_env_shape


# TODO: make this a class, to reduce number of arguments passed
# TODO: can this share more structure with plan & execute?
def collect_trajectory(scenario: ExperimentScenario,
                       params: Dict,
                       service_provider,
                       traj_idx: int,
                       action_rng: np.random.RandomState,
                       verbose: int,
                       ):
    if params['no_objects']:
        rows, cols, channels = extent_to_env_shape(params['extent'], params['res'])
        origin = np.array([rows // 2, cols // 2, channels // 2], dtype=np.int32)
        env = np.zeros([rows, cols, channels], dtype=np.float32)
        environment = {'env': env, 'res': params['res'], 'origin': origin, 'extent': params['extent']}
    else:
        # At this point, we hope all of the objects have stopped moving, so we can get the environment and assume it never changes
        # over the course of this function
        environment = ros_pycommon.get_environment_for_extents_3d(extent=params['extent'],
                                                                  res=params['res'],
                                                                  service_provider=service_provider,
                                                                  robot_name=scenario.robot_name())

    feature = dict_of_float_tensors_to_bytes_feature(environment)
    feature['traj_idx'] = float_tensor_to_bytes_feature(traj_idx)

    # Visualization
    scenario.plot_environment_rviz(environment)
    scenario.plot_traj_idx_rviz(traj_idx)

    actions = {k: [] for k in scenario.actions_description().keys()}
    states = {k: [] for k in scenario.states_description().keys()}

    # sanity check!
    for k in scenario.actions_description().keys():
        if k in scenario.states_description().keys():
            rospy.logerr(f"Duplicate key {k} is both a state and an action")

    time_indices = []
    for time_idx in range(params['steps_per_traj']):
        # get current state and sample action
        state = scenario.get_state()
        action = scenario.sample_action(environment=environment,
                                        state=state,
                                        data_collection_params=params,
                                        action_params=params,
                                        action_rng=action_rng)

        # execute action
        scenario.execute_action(action)

        # add to the dataset
        if time_idx < params['steps_per_traj'] - 1:  # skip the last action
            for action_name in scenario.actions_description().keys():
                action_component = action[action_name]
                actions[action_name].append(action_component)
        for state_component_name in scenario.states_description().keys():
            state_component = state[state_component_name]
            states[state_component_name].append(state_component)
        time_indices.append(time_idx)

        # Visualization
        scenario.plot_state_rviz(state, label='actual')
        if time_idx < params['steps_per_traj'] - 1:  # skip the last action in visualization as well
            scenario.plot_action_rviz(state, action)
        scenario.plot_time_idx_rviz(time_idx)

    feature.update(dict_of_float_tensors_to_bytes_feature(states))
    feature.update(dict_of_float_tensors_to_bytes_feature(actions))
    feature['time_idx'] = float_tensor_to_bytes_feature(time_indices)

    if verbose:
        print(Fore.GREEN + "Trajectory {} Complete".format(traj_idx) + Fore.RESET)

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    example = example_proto.SerializeToString()
    return example


def generate_trajs(service_provider,
                   scenario: ExperimentScenario,
                   params: Dict,
                   args,
                   full_output_directory,
                   env_rng: np.random.RandomState,
                   action_rng: np.random.RandomState,
                   ):
    examples = np.ndarray([params['trajs_per_file']], dtype=object)
    last_record_t = perf_counter()

    for traj_idx in range(args.trajs):
        # Randomize the environment
        if not params['no_objects'] and traj_idx % params["randomize_environment_every_n_trajectories"] == 0:
            scenario.randomize_environment(env_rng, objects_params=params, data_collection_params=params)

        # Generate a new trajectory
        example = collect_trajectory(scenario=scenario,
                                     params=params,
                                     service_provider=service_provider,
                                     traj_idx=traj_idx,
                                     action_rng=action_rng,
                                     verbose=args.verbose)
        current_record_traj_idx = traj_idx % params['trajs_per_file']
        examples[current_record_traj_idx] = example

        # Save the data
        if current_record_traj_idx == params['trajs_per_file'] - 1:
            # Construct the dataset where each trajectory has been serialized into one big string
            # since TFRecords don't really support hierarchical data structures
            serialized_dataset = tf.data.Dataset.from_tensor_slices((examples))

            end_traj_idx = traj_idx + args.start_idx_offset
            start_traj_idx = end_traj_idx - params['trajs_per_file'] + 1
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


def generate(service_provider, params: Dict, args):
    rospy.init_node('collect_dynamics_data')
    scenario = get_scenario(args.scenario)

    assert args.trajs % params['trajs_per_file'] == 0, f"num trajs must be multiple of {params['trajs_per_file']}"

    # full_output_directory = data_directory(args.outdir, args.trajs)
    print("USING OUTDIR EXACTLY")
    full_output_directory = pathlib.Path(args.outdir)
    full_output_directory.mkdir(exist_ok=True)

    if not os.path.isdir(full_output_directory) and args.verbose:
        print(Fore.YELLOW + "Creating output directory: {}".format(full_output_directory) + Fore.RESET)
        os.mkdir(full_output_directory)

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
    print(Fore.CYAN + "Using seed: {}".format(args.seed) + Fore.RESET)

    with open(pathlib.Path(full_output_directory) / 'hparams.json', 'w') as of:
        options = {
            'seed': args.seed,
            'n_trajs': args.trajs,
            'data_collection_params': params,
            'states_description': scenario.states_description(),
            'action_description': scenario.actions_description(),
            'scenario': args.scenario,
        }
        json.dump(options, of, indent=2)

    np.random.seed(args.seed)
    env_rng = np.random.RandomState(args.seed)
    action_rng = np.random.RandomState(args.seed)

    service_provider.setup_env(verbose=args.verbose,
                               real_time_rate=args.real_time_rate,
                               max_step_size=params['max_step_size'])

    generate_trajs(service_provider=service_provider,
                   scenario=scenario,
                   params=params,
                   args=args,
                   full_output_directory=full_output_directory,
                   env_rng=env_rng,
                   action_rng=action_rng)
