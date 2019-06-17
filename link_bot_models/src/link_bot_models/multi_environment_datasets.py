import json
import os

import numpy as np

from link_bot_models.label_types import LabelType
from link_bot_pycommon.link_bot_pycommon import SDF


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

    def __init__(self, filename_pairs, constraint_label_types, n_obstacles, obstacle_size, threshold, seed):
        self.constraint_label_types = constraint_label_types
        self.n_obstacles = n_obstacles
        self.obstacle_size = obstacle_size
        self.threshold = threshold
        self.seed = seed

        # convert all file paths to be absolute
        self.abs_filename_pairs = []
        for filename_pair in filename_pairs:
            self.abs_filename_pairs.append([os.path.abspath(filename) for filename in filename_pair])

        # construct a flat list of training examples which are distributed in the files listed in the filename_pairs
        self.example_information = []
        self.sdf_shape = None
        self.n_environments = len(filename_pairs)
        example_id = 0
        for i, (sdf_filename, rope_data_filename) in enumerate(filename_pairs):
            sdf_data = SDF.load(sdf_filename)
            rope_data = np.load(rope_data_filename)
            examples_in_env = rope_data['rope_configurations'].shape[0]

            if i == 0:
                # store useful information about the dataset
                self.rope_configurations_per_env = examples_in_env
                self.sdf_shape = sdf_data.sdf.shape
                self.sdf_resolution = sdf_data.resolution.tolist()
            error_msg = "SDFs shape {} doesn't match shape {}".format(self.sdf_shape, sdf_data.sdf.shape)
            assert self.sdf_shape == sdf_data.sdf.shape, error_msg

            for j in range(examples_in_env):
                self.example_information.append({
                    'sdf_filename': sdf_filename,
                    'rope_data_filename': rope_data_filename,
                    'rope_data_index': j
                })
                example_id += 1
        self.num_examples = example_id

    def generator(self, batch_size):
        return DatasetGenerator(self, self.constraint_label_types, batch_size)

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


class DatasetGenerator:

    def __init__(self, dataset, label_types, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        examples_ids = np.arange(0, self.dataset.num_examples)
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
        x = {
            'sdf': np.ndarray([self.batch_size, self.dataset.sdf_shape[0], self.dataset.sdf_shape[1]]),
            'sdf_gradient': np.ndarray([self.batch_size, self.dataset.sdf_shape[0], self.dataset.sdf_shape[1], 2]),
            'sdf_origin': np.ndarray([self.batch_size, 2]),
            'sdf_resolution': np.ndarray([self.batch_size, 2]),
            'sdf_extent': np.ndarray([self.batch_size, 4]),
            'rope_configuration': np.ndarray([self.batch_size, 6])
        }
        y = {
            'combined_output': np.ndarray([self.batch_size, 1]),
            'all_output': np.ndarray([self.batch_size, self.label_mask.shape[0]]),
        }

        for i, example_id in enumerate(batch_indeces):
            example_info = self.dataset.example_information[example_id]
            sdf_data = SDF.load(example_info['sdf_filename'])
            rope_data = np.load(example_info['rope_data_filename'])

            rope_configuration = rope_data['rope_configurations'][example_info['rope_data_index']]

            all_label = rope_data['constraints'][example_info['rope_data_index']].astype(np.float32)
            combined_label = np.any(all_label * self.label_mask).astype(np.float32)

            x['sdf'][i] = sdf_data.sdf
            x['sdf_gradient'][i] = sdf_data.gradient
            x['sdf_origin'][i] = sdf_data.origin
            x['sdf_resolution'][i] = sdf_data.resolution
            x['sdf_extent'][i] = sdf_data.extent
            x['rope_configuration'][i] = rope_configuration

            y['all_output'][i] = all_label
            y['combined_output'][i] = combined_label

        return x, y
