import json
import os

import numpy as np

from link_bot_pycommon.link_bot_pycommon import SDF


class Environment:

    def __init__(self, sdf_data, rope_data):
        self.sdf_data = sdf_data
        self.rope_data = rope_data


class MultiEnvironmentDataset:

    def __init__(self, filename_pairs, n_obstacles, obstacle_size, threshold):
        self.n_obstacles = n_obstacles
        self.obstacle_size = obstacle_size
        self.threshold = threshold

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

            env = Environment(sdf_data, rope_data)
            self.environments[i] = env

    @staticmethod
    def load_dataset(dataset_filename):
        dataset_dict = json.load(open(dataset_filename, 'r'))
        filename_pairs = dataset_dict['filename_pairs']
        n_obstacles = dataset_dict['n_obstacles']
        obstacle_size = dataset_dict['obstacle_size']
        threshold = dataset_dict['threshold']
        dataset = MultiEnvironmentDataset(filename_pairs, n_obstacles=n_obstacles, obstacle_size=obstacle_size,
                                          threshold=threshold)
        return dataset

    def save(self, dataset_filename):
        dataset_dict = {
            'n_environments': len(self.abs_filename_pairs),
            'sdf_shape': self.sdf_shape,
            'sdf_resolution': self.sdf_resolution,
            'n_rope_configurations_per_env': self.rope_configurations_per_env,
            'n_obstacles': self.n_obstacles,
            'obstacle_size': self.obstacle_size,
            'threshold': self.threshold,
            'filename_pairs': self.abs_filename_pairs,
        }
        json.dump(dataset_dict, open(dataset_filename, 'w'), sort_keys=True, indent=4)
