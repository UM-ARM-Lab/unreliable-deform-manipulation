#!/usr/bin/env python
import json
import os
import pathlib
import sys
from time import perf_counter

import numpy as np
import tensorflow as tf
from colorama import Fore

import rospy
from link_bot_data.link_bot_dataset_utils import float_tensor_to_bytes_feature, data_directory
from link_bot_pycommon import ros_pycommon
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.link_bot_sdf_utils import OccupancyData, env_from_occupancy_data
from link_bot_pycommon.params import FullEnvParams, SimParams, CollectDynamicsParams
from link_bot_pycommon.ros_pycommon import get_states_dict


def generate_traj(scenario: ExperimentScenario,
                  params: CollectDynamicsParams,
                  service_provider,
                  traj_idx: int,
                  global_t_step: int,
                  action_rng: np.random.RandomState,
                  verbose: int):
    if params.no_obstacles:
        rows = int(params.full_env_h_m // params.res)
        cols = int(params.full_env_w_m // params.res)
        full_env_origin = np.array([rows // 2, cols // 2], dtype=np.int32)
        data = np.zeros([rows, cols], dtype=np.float32)
        full_env_data = OccupancyData(data, params.res, full_env_origin)
    else:
        # At this point, we hope all of the objects have stopped moving, so we can get the environment and assume it never changes
        # over the course of this function
        full_env_data = ros_pycommon.get_occupancy_data(env_w_m=params.full_env_w_m,
                                                        env_h_m=params.full_env_h_m,
                                                        res=params.res,
                                                        service_provider=service_provider,
                                                        robot_name=scenario.robot_name())

    feature = {
        'full_env/env': float_tensor_to_bytes_feature(full_env_data.data),
        'full_env/extent': float_tensor_to_bytes_feature(full_env_data.extent),
        'full_env/origin': float_tensor_to_bytes_feature(full_env_data.origin),
        'full_env/res': float_tensor_to_bytes_feature(full_env_data.resolution),
    }

    action_msg = None
    for time_idx in range(params.steps_per_traj):
        state_dict = get_states_dict(service_provider)
        environment = env_from_occupancy_data(full_env_data)
        action_msg = scenario.sample_action(environment=environment,
                                            service_provider=service_provider,
                                            state=state_dict,
                                            last_action=action_msg,
                                            params=params,
                                            action_rng=action_rng)

        feature['{}/action'.format(time_idx)] = float_tensor_to_bytes_feature(action_msg.action)
        for name, state_dict in state_dict.items():
            feature['{}/{}'.format(time_idx, name)] = float_tensor_to_bytes_feature(state_dict)
        feature['{}/traj_idx'.format(time_idx)] = float_tensor_to_bytes_feature(traj_idx)
        feature['{}/time_idx'.format(time_idx)] = float_tensor_to_bytes_feature(time_idx)

        service_provider.execute_action(action_msg)

        global_t_step += 1

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
                   action_rng: np.random.RandomState):
    examples = np.ndarray([params.trajs_per_file], dtype=object)
    global_t_step = 0
    last_record_t = perf_counter()
    for traj_idx in range(args.trajs):
        if params.reset_robot is not None or params.reset_world:
            service_provider.reset_world(args.verbose, reset_robot=params.reset_robot)

        # Might not do anything, depends on args
        rearrange_environment(service_provider, params, env_rng)

        # Generate a new trajectory
        example, global_t_step = generate_traj(scenario=scenario,
                                               params=params,
                                               service_provider=service_provider,
                                               traj_idx=traj_idx,
                                               global_t_step=global_t_step,
                                               action_rng=action_rng,
                                               verbose=args.verbose)
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

    full_env_rows = int(params.full_env_h_m / params.res)
    full_env_cols = int(params.full_env_w_m / params.res)
    full_env_params = FullEnvParams(h_rows=full_env_rows, w_cols=full_env_cols, res=params.res)
    sim_params = SimParams(real_time_rate=args.real_time_rate,
                           max_step_size=params.max_step_size,
                           movable_obstacles=params.movable_obstacles)

    states_description = service_provider.get_states_description()
    n_action = service_provider.get_n_action()

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
    print(Fore.CYAN + "Using seed: {}".format(args.seed) + Fore.RESET)

    with open(pathlib.Path(full_output_directory) / 'hparams.json', 'w') as of:
        options = {
            'seed': args.seed,
            'dt': params.dt,
            'n_trajs': args.trajs,
            'max_step_size': params.max_step_size,
            'full_env_params': full_env_params.to_json(),
            'sim_params': sim_params.to_json(),
            'sequence_length': params.steps_per_traj,
            'states_description': states_description,
            'n_action': n_action,
            'scenario': args.scenario,
        }
        json.dump(options, of, indent=2)

    np.random.seed(args.seed)
    env_rng = np.random.RandomState(args.seed)
    action_rng = np.random.RandomState(args.seed)

    service_provider.setup_env(verbose=args.verbose,
                               real_time_rate=args.real_time_rate,
                               max_step_size=params.max_step_size,
                               reset_robot=[0, 0])

    generate_trajs(service_provider, scenario, params, args, full_output_directory, env_rng, action_rng)
