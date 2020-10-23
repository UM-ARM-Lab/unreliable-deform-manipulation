#!/usr/bin/env python
import json
import pathlib
from time import perf_counter
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from colorama import Fore

import rospy
from link_bot_data.files_dataset import FilesDataset
from link_bot_data.link_bot_dataset_utils import data_directory, dict_of_float_tensors_to_bytes_feature
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.grid_utils import extent_to_env_shape


class DataCollector:

    def __init__(self,
                 scenario_name: str,
                 service_provider: BaseServices,
                 params: Dict,
                 seed: Optional[int] = None,
                 verbose: int = 0):
        self.service_provider = service_provider
        self.params = params
        self.verbose = verbose
        self.scenario_name = scenario_name
        self.scenario = get_scenario(scenario_name)

        if seed is None:
            self.seed = np.random.randint(0, 100)
        else:
            self.seed = seed
        print(Fore.CYAN + f"Using seed: {self.seed}" + Fore.RESET)

        service_provider.setup_env(verbose=self.verbose,
                                   real_time_rate=1.0,
                                   max_step_size=self.params['max_step_size'])

    def collect_trajectory(self,
                           traj_idx: int,
                           verbose: int,
                           action_rng: np.random.RandomState
                           ):
        if self.params['no_objects']:
            rows, cols, channels = extent_to_env_shape(self.params['extent'], self.params['res'])
            origin = np.array([rows // 2, cols // 2, channels // 2], dtype=np.int32)
            env = np.zeros([rows, cols, channels], dtype=np.float32)
            environment = {'env': env, 'res': self.params['res'], 'origin': origin, 'extent': self.params['extent']}
        else:
            # At this point, we hope all of the objects have stopped moving, so we can get the environment and assume it never changes
            # over the course of this function
            environment = self.scenario.get_environment(self.params)

        feature = environment
        feature['traj_idx'] = traj_idx

        # Visualization
        self.scenario.plot_environment_rviz(environment)
        self.scenario.plot_traj_idx_rviz(traj_idx)

        actions = {k: [] for k in self.scenario.actions_description().keys()}
        states = {k: [] for k in self.scenario.state_like_keys()}

        # sanity check!
        for k in self.scenario.actions_description().keys():
            if k in self.scenario.states_description().keys():
                rospy.logerr(f"Duplicate key {k} is both a state and an action")

        time_indices = []
        for time_idx in range(self.params['steps_per_traj']):
            # get current state and sample action
            state = self.scenario.get_state()
            # TODO: sample the entire action sequence in advance?
            action = self.scenario.sample_action(action_rng=action_rng,
                                                 environment=environment,
                                                 state=state,
                                                 action_params=self.params)

            # execute action
            self.scenario.execute_action(action)

            # add to the dataset
            if time_idx < self.params['steps_per_traj'] - 1:  # skip the last action
                for action_name in self.scenario.actions_description().keys():
                    action_component = action[action_name]
                    actions[action_name].append(action_component)
            state_like_keys = self.scenario.state_like_keys()
            for state_component_name in state_like_keys:
                state_component = state[state_component_name]
                states[state_component_name].append(state_component)
            time_indices.append(time_idx)

            # Visualization
            self.scenario.plot_state_rviz(state, label='actual')
            if time_idx < self.params['steps_per_traj'] - 1:  # skip the last action in visualization as well
                self.scenario.plot_action_rviz(state, action)
            self.scenario.plot_time_idx_rviz(time_idx)

        feature.update(states)
        feature.update(actions)
        feature['time_idx'] = time_indices

        if verbose:
            print(Fore.GREEN + "Trajectory {} Complete".format(traj_idx) + Fore.RESET)

        return feature

    def collect_data(self,
                     n_trajs: int,
                     nickname: str,
                     robot_namespace: str,
                     ):
        outdir = pathlib.Path('fwd_model_data') / nickname
        full_output_directory = data_directory(outdir, n_trajs)

        files_dataset = FilesDataset(full_output_directory)

        full_output_directory.mkdir(exist_ok=True)
        print(Fore.GREEN + full_output_directory.as_posix() + Fore.RESET)

        dataset_hparams = {
            'nickname':                         nickname,
            'robot_namespace':                  robot_namespace,
            'seed':                             self.seed,
            'n_trajs':                          n_trajs,
            'data_collection_params':           self.params,
            'states_description':               self.scenario.states_description(),
            'action_description':               self.scenario.actions_description(),
            'observations_description':         self.scenario.observations_description(),
            'observation_features_description': self.scenario.observation_features_description(),
            'scenario':                         self.scenario_name,
            'scenario_metadata':                self.scenario.dynamics_dataset_metadata(),
        }
        with (full_output_directory / 'hparams.json').open('w') as dataset_hparams_file:
            json.dump(dataset_hparams, dataset_hparams_file, indent=2)
        record_options = tf.io.TFRecordOptions(compression_type='ZLIB')

        self.scenario.randomization_initialization()
        self.scenario.on_before_data_collection(self.params)

        t0 = perf_counter()

        combined_seeds = [traj_idx + 100000 * self.seed for traj_idx in range(n_trajs)]
        for traj_idx, seed in enumerate(combined_seeds):
            # combine the trajectory idx and the overall "seed" to make a unique seed for each trajectory/seed pair
            env_rng = np.random.RandomState(seed)
            action_rng = np.random.RandomState(seed)

            # Randomize the environment
            randomize = self.params["randomize_n"] and traj_idx % self.params["randomize_n"] == 0
            if not self.params['no_objects'] and randomize:
                self.scenario.randomize_environment(env_rng,
                                                    objects_params=self.params,
                                                    data_collection_params=self.params)

            # Generate a new trajectory
            example = self.collect_trajectory(traj_idx=traj_idx, verbose=self.verbose, action_rng=action_rng)
            print(f'traj {traj_idx}/{n_trajs} ({seed}), {perf_counter() - t0:.4f}s')

            # Save the data
            features = dict_of_float_tensors_to_bytes_feature(example)
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))
            example_str = example_proto.SerializeToString()
            full_filename = full_output_directory / f"example_{traj_idx:09d}.tfrecords"
            files_dataset.add(full_filename)
            with tf.io.TFRecordWriter(str(full_filename), record_options) as writer:
                writer.write(example_str)

        self.scenario.on_after_data_collection(self.params)

        print(Fore.GREEN + full_output_directory.as_posix() + Fore.RESET)

        return files_dataset
