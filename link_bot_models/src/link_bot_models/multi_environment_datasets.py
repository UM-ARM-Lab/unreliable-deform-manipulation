import json
import os
from enum import auto

import numpy as np

from link_bot_pycommon import link_bot_pycommon
from link_bot_pycommon.link_bot_pycommon import SDF


class LabelType(link_bot_pycommon.ArgsEnum):
    SDF = auto()
    Overstretching = auto()
    SDF_and_Overstretching = auto()


class FancyList:

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return self.items[0].shape[0]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.items[index]
        elif isinstance(index, list):
            return [self.items[i] for i in index]
        elif isinstance(index, slice):
            return self.items[index]
        elif isinstance(index, tuple):
            if isinstance(index[0], slice) and isinstance(index[1], int):
                return [item[index[1]] for item in self.items[index[0]]]
            elif isinstance(index[0], slice) and isinstance(index[1], list):
                return [[item[j] for j in index[1]] for item in self.items[index[0]]]
            elif isinstance(index[0], slice) and isinstance(index[1], np.ndarray) and len(index[1].shape) == 1:
                return [[item[j] for j in index[1]] for item in self.items[index[0]]]
            else:
                raise ValueError("unsupported types for index: (" + ', '.join([str(type(i)) for i in index]) + ")")
        else:
            raise ValueError("unsupported type for index {}".format(type(index)))


def make_inputs_and_labels(environments):
    """ merges all the data and labels from the environments into a format suitable for batching into feed dicts """
    sdfs = []
    sdf_gradients = []
    sdf_origins = []
    sdf_resolutions = []
    sdf_extents = []
    rope_configurations = []
    constraint_labels = []
    for environment in environments:
        # environment is a multi_environment_datasets.Environment
        env_rope_configurations = environment.rope_data['rope_configurations']
        env_constraint_labels = environment.rope_data['constraints']
        for rope_configuration, constraint_label in zip(env_rope_configurations, env_constraint_labels):
            sdfs.append(environment.sdf_data.sdf)
            sdf_gradients.append(environment.sdf_data.gradient)
            sdf_origins.append(environment.sdf_data.origin)
            sdf_resolutions.append(environment.sdf_data.resolution)
            sdf_extents.append(environment.sdf_data.extent)
            rope_configurations.append(rope_configuration)
            constraint_labels.append(constraint_label)

    inputs = FancyList([
        np.array(rope_configurations),
        np.array(sdfs),
        np.array(sdf_gradients),
        np.array(sdf_origins),
        np.array(sdf_resolutions),
        np.array(sdf_extents),
    ])
    labels = FancyList([
        np.array(constraint_labels),
    ])

    return inputs, labels


class EnvironmentData:

    def __init__(self, sdf_data, rope_data):
        self.sdf_data = sdf_data
        self.rope_data = rope_data


class MultiEnvironmentDataset:

    def __init__(self, filename_pairs, n_obstacles, obstacle_size, threshold, seed):
        self.n_obstacles = n_obstacles
        self.obstacle_size = obstacle_size
        self.threshold = threshold
        self.seed = seed

        self.abs_filename_pairs = []
        for filename_pair in filename_pairs:
            self.abs_filename_pairs.append([os.path.abspath(filename) for filename in filename_pair])

        # make sure all the SDFs have the same shape
        self.sdf_shape = None
        self.n_environments = len(filename_pairs)
        self.environments = np.ndarray([self.n_environments], dtype=np.object)
        for i, (sdf_filename, rope_data_filename) in enumerate(filename_pairs):
            sdf_data = SDF.load(sdf_filename)
            rope_data = np.load(rope_data_filename)

            if i == 0:
                self.rope_configurations_per_env = rope_data['rope_configurations'].shape[0]
                self.sdf_shape = sdf_data.sdf.shape
                self.sdf_resolution = sdf_data.resolution.tolist()
            error_msg = "SDFs shape {} doesn't match shape {}".format(self.sdf_shape, sdf_data.sdf.shape)
            assert self.sdf_shape == sdf_data.sdf.shape, error_msg

            env = EnvironmentData(sdf_data, rope_data)
            self.environments[i] = env

    @staticmethod
    def load_dataset(dataset_filename):
        dataset_dict = json.load(open(dataset_filename, 'r'))
        filename_pairs = dataset_dict['filename_pairs']
        n_obstacles = dataset_dict['n_obstacles']
        obstacle_size = dataset_dict['obstacle_size']
        threshold = dataset_dict['threshold']
        seed = dataset_dict['seed']
        dataset = MultiEnvironmentDataset(filename_pairs, n_obstacles=n_obstacles, obstacle_size=obstacle_size,
                                          threshold=threshold, seed=seed)
        return dataset

    def save(self, dataset_filename):
        dataset_dict = {
            'n_environments': len(self.abs_filename_pairs),
            'seed': self.seed,
            'sdf_shape': self.sdf_shape,
            'sdf_resolution': self.sdf_resolution,
            'n_rope_configurations_per_env': self.rope_configurations_per_env,
            'n_obstacles': self.n_obstacles,
            'obstacle_size': self.obstacle_size,
            'threshold': self.threshold,
            'filename_pairs': self.abs_filename_pairs,
        }
        json.dump(dataset_dict, open(dataset_filename, 'w'), sort_keys=True, indent=4)
