import json
import os

import keras
import numpy as np

from link_bot_models.label_types import LabelType
from link_bot_pycommon import link_bot_pycommon
from link_bot_pycommon.link_bot_sdf_tools import SDF


class EnvironmentData:

    def __init__(self, sdf_data, rope_data):
        self.sdf_data = sdf_data
        self.rope_data = rope_data


class MultiEnvironmentDataset:

    def __init__(self, filename_pairs, constraint_label_types, n_obstacles, obstacle_size, threshold, seed):
        self.constraint_label_types = constraint_label_types
        self.n_obstacles = n_obstacles
        self.obstacle_size = obstacle_size
        self.threshold = threshold
        self.seed = seed
        self.N = 0

        # convert all file paths to be absolute
        self.abs_filename_pairs = []
        for filename_pair in filename_pairs:
            self.abs_filename_pairs.append([os.path.abspath(filename) for filename in filename_pair])

        # construct a flat list of training examples which are distributed in the files listed in the filename_pairs
        self.example_information = []
        self.sdf_shape = None
        self.n_environments = len(filename_pairs)
        self.environments = np.ndarray([self.n_environments], dtype=np.object)
        example_id = 0
        for i, (sdf_data_filename, rope_data_filename) in enumerate(filename_pairs):
            sdf_data = SDF.load(sdf_data_filename)
            rope_data = np.load(rope_data_filename)
            examples_in_env, N = rope_data['rope_configurations'].shape

            if i == 0:
                # store useful information about the dataset
                self.rope_configurations_per_env = examples_in_env
                self.N = N
                self.sdf_shape = sdf_data.sdf.shape
                self.sdf_resolution = sdf_data.resolution.tolist()
            error_msg = "SDFs shape {} doesn't match shape {}".format(self.sdf_shape, sdf_data.sdf.shape)
            assert self.sdf_shape == sdf_data.sdf.shape, error_msg

            env = EnvironmentData(sdf_data, rope_data)
            self.environments[i] = env

            for j in range(examples_in_env):
                self.example_information.append({
                    'sdf_data_filename': sdf_data_filename,
                    'rope_data_filename': rope_data_filename,
                    'rope_data_index': j
                })
                example_id += 1
        self.num_examples = example_id

    def generator(self, batch_size):
        return self.generator_specific_labels(self.constraint_label_types, batch_size)

    def generator_specific_labels(self, constraint_label_types, batch_size):
        """ allows you to test with just some of the labels """
        return DatasetGenerator(self, constraint_label_types, batch_size)

    @staticmethod
    def load_dataset(dataset_filename):
        dataset_dict = json.load(open(dataset_filename, 'r'))
        constraint_label_types = [LabelType[label_type] for label_type in dataset_dict['constraint_label_types']]
        filename_pairs = dataset_dict['filename_pairs']
        n_obstacles = dataset_dict['n_obstacles']
        obstacle_size = dataset_dict['obstacle_size']
        threshold = dataset_dict['threshold']
        seed = dataset_dict['seed']
        dataset = MultiEnvironmentDataset(filename_pairs, constraint_label_types=constraint_label_types, n_obstacles=n_obstacles,
                                          obstacle_size=obstacle_size,
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
            'constraint_label_types': [label_type.name for label_type in self.constraint_label_types],
        }
        json.dump(dataset_dict, open(dataset_filename, 'w'), sort_keys=True, indent=4)

    def __len__(self):
        return self.num_examples


class DatasetGenerator(keras.utils.Sequence):

    def __init__(self, dataset, label_types, batch_size, shuffle=True):
        batch_size_err_msg = "Batch size {} must even divide the dataset size {}".format(batch_size, len(dataset))
        assert len(dataset) % batch_size == 0, batch_size_err_msg

        self.dataset = dataset
        self.batch_size = batch_size
        examples_ids = np.arange(0, self.dataset.num_examples)
        if shuffle:
            np.random.shuffle(examples_ids)
        self.batches = np.reshape(examples_ids, [-1, batch_size])
        self.label_mask = LabelType.mask(label_types)

    def __len__(self):
        return self.batches.shape[0]

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: an id number for which batch to load
        :return: a batch of data (x, y) where x and y are numpy arrays
        """
        batch_indeces = self.batches[index]
        n_rope_points = int(self.dataset.N // 2)
        x = {
            'sdf': np.ndarray([self.batch_size, self.dataset.sdf_shape[0], self.dataset.sdf_shape[1], 1]),
            'sdf_gradient': np.ndarray([self.batch_size, self.dataset.sdf_shape[0], self.dataset.sdf_shape[1], 2]),
            'sdf_origin': np.ndarray([self.batch_size, 2]),
            'sdf_resolution': np.ndarray([self.batch_size, 2]),
            'sdf_extent': np.ndarray([self.batch_size, 4]),
            'rope_configuration': np.ndarray([self.batch_size, self.dataset.N]),
            'rope_image': np.ndarray([self.batch_size, self.dataset.sdf_shape[0], self.dataset.sdf_shape[1], n_rope_points]),
        }
        y = {
            'combined_output': np.ndarray([self.batch_size, 1]),
            'all_output': np.ndarray([self.batch_size, self.label_mask.shape[0]]),
        }

        loaded_data_cache = {}
        for i, example_id in enumerate(batch_indeces):
            example_info = self.dataset.example_information[example_id]

            sdf_data_filename = example_info['sdf_data_filename']
            rope_data_filename = example_info['rope_data_filename']

            if sdf_data_filename in loaded_data_cache:
                sdf_data = loaded_data_cache[sdf_data_filename]
            else:
                sdf_data = SDF.load(sdf_data_filename)
                loaded_data_cache[sdf_data_filename] = sdf_data

            if rope_data_filename in loaded_data_cache:
                rope_data = loaded_data_cache[rope_data_filename]
            else:
                rope_data = np.load(rope_data_filename)
                loaded_data_cache[rope_data_filename] = rope_data

            rope_configuration = rope_data['rope_configurations'][example_info['rope_data_index']]

            rope_image = link_bot_pycommon.make_rope_images(sdf_data, rope_configuration)

            all_label = rope_data['constraints'][example_info['rope_data_index']].astype(np.float32)
            combined_label = np.any(all_label * self.label_mask).astype(np.float32)

            x['sdf'][i] = np.atleast_3d(sdf_data.sdf)
            x['sdf_gradient'][i] = sdf_data.gradient
            x['sdf_origin'][i] = sdf_data.origin
            x['sdf_resolution'][i] = sdf_data.resolution
            x['sdf_extent'][i] = sdf_data.extent
            x['rope_configuration'][i] = rope_configuration
            x['rope_image'][i] = rope_image

            y['all_output'][i] = all_label
            y['combined_output'][i] = combined_label

        return x, y
