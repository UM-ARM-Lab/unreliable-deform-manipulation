import pathlib
from typing import List

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

        self.state_feature_names = [
            'time_idx',
        ]
        self.states_description = self.hparams['states_description']
        for k in self.states_description.keys():
            self.state_feature_names.append('{}'.format(k))

        self.action_feature_names = ['delta_position']
        # self.action_description = self.hparams['action_description']
        # for k in self.action_description.keys():
        #     self.action_feature_names.append('{}'.format(k))

        self.constant_feature_names = [
            'env',
            'extent',
            'origin',
            'res',
            'traj_idx',
        ]

        self.int64_keys = ['time_idx']

    def make_features_description(self):
        features_description = {}
        for feature_name in self.constant_feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        for feature_name in self.state_feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)
        for feature_name in self.action_feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        return features_description

    def index_time(self, example, t):
        """ index the time-based features. assumes batch """
        example_t = {}
        for feature_name in self.constant_feature_names:
            example_t[feature_name] = example[feature_name]

        for feature_name in self.state_feature_names:
            example_t[feature_name] = example[feature_name][:, t]

        for feature_name in self.action_feature_names:
            example_t[feature_name] = example[feature_name][:, t]

        return example_t

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        def _make_time_int(example):
            example['time_idx'] = tf.cast(example['time_idx'], tf.int64)
            return example

        def _drop_last_action(example):
            for k in self.action_feature_names:
                example[k] = example[k][:-1]
            return example

        dataset = dataset.map(_make_time_int)
        dataset = dataset.map(_drop_last_action)
        return dataset
