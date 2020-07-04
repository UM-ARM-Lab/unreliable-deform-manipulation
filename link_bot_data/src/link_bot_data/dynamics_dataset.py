import pathlib
from typing import List, Dict

import tensorflow as tf

from link_bot_data.base_dataset import BaseDataset
from link_bot_pycommon.get_scenario import get_scenario


class DynamicsDataset(BaseDataset):
    def __init__(self, dataset_dirs: List[pathlib.Path], step_size: int = 1):
        """
        :param dataset_dirs: dataset directories
        :param step_size: the number of time steps to skip when slicing the full trajectories into trajectories for training
        """
        super(DynamicsDataset, self).__init__(dataset_dirs)

        self.step_size = step_size
        self.scenario = get_scenario(self.hparams['scenario'])

        self.state_keys = list(self.hparams['states_description'].keys())
        self.state_keys.append('time_idx')

        self.action_keys = list(self.hparams['action_description'].keys())

        self.constant_feature_names = [
            'env',
            'extent',
            'origin',
            'res',
            'traj_idx',
        ]

        self.int64_keys = ['time_idx']

        self.data_collection_params = self.hparams['data_collection_params']
        self.sequence_length = self.hparams['data_collection_params']['steps_per_traj']

    def make_features_description(self):
        features_description = {}
        for feature_name in self.constant_feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        for feature_name in self.state_keys:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)
        for feature_name in self.action_keys:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        return features_description

    def index_time(self, example, t):
        """ index the time-based features. assumes batch """
        example_t = {}
        for feature_name in self.constant_feature_names:
            example_t[feature_name] = example[feature_name]

        for feature_name in self.state_keys:
            example_t[feature_name] = example[feature_name][:, t]

        for feature_name in self.action_keys:
            if t < example[feature_name].shape[1]:
                example_t[feature_name] = example[feature_name][:, t]
            else:
                example_t[feature_name] = example[feature_name][:, t - 1]

        return example_t

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        # def _make_time_int(example: Dict):
        #     example['time_idx'] = tf.cast(example['time_idx'], tf.int64)
        #     return example

        # def _drop_last_action(example: Dict):
        #     for k in self.action_keys:
        #         example[k] = example[k][:-1]
        #     return example

        # dataset = dataset.map(_make_time_int)

        # def _add_time(example: Dict):
        #     # this function is called before batching occurs, so the first dimension should be time
        #     example['time'] = example[self.state_keys[0]].shape[0]
        #     return example

        # dataset = dataset.map(_add_time)
        return dataset
