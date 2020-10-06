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

        self.observation_feature_keys = list(self.hparams['observation_feature_description'].keys())

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
        if 'new_sequence_length' in self.hparams:
            self.sequence_length = self.hparams['new_sequence_length']
        else:
            self.sequence_length = self.hparams['data_collection_params']['steps_per_traj']
        self.batch_metadata = {
            'sequence_length': self.sequence_length
        }

    def make_features_description(self):
        features_description = {}
        for feature_name in self.constant_feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        for feature_name in self.state_keys:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)
        for feature_name in self.observation_feature_keys:
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

        scenario_metadata = self.hparams['scenario_metadata']
        for k in scenario_metadata.keys():
            example_t[k] = example[k]

        return example_t

    def split_into_sequences(self, example, desired_sequence_length):
        # return a dict where every element has different sequences split across the 0th dimension
        for start_t in range(0, self.sequence_length - desired_sequence_length + 1, desired_sequence_length):
            out_example = {}
            for k in self.constant_feature_names:
                out_example[k] = example[k]

            for k in self.state_keys:
                v = example[k][start_t:start_t + desired_sequence_length]
                assert v.shape[0] == desired_sequence_length
                out_example[k] = v

            for k in self.action_keys:
                v = example[k][start_t:start_t + desired_sequence_length - 1]
                assert v.shape[0] == (desired_sequence_length - 1)
                out_example[k] = v

            yield out_example

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int, **kwargs):
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

        scenario_metadata = self.hparams['scenario_metadata']

        def _add_scenario_metadata(example: Dict):
            example.update(scenario_metadata)
            return example

        dataset = dataset.map(_add_scenario_metadata)


        return dataset
