import pathlib
from typing import List, Dict, Optional

import tensorflow as tf
from colorama import Fore

from link_bot_data.base_dataset import BaseDatasetLoader
from link_bot_data.dataset_utils import use_gt_rope
from moonshine.indexing import index_time_batched
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.moonshine_utils import numpify, remove_batch


class DynamicsDatasetLoader(BaseDatasetLoader):
    def __init__(self, dataset_dirs: List[pathlib.Path], step_size: int = 1, use_gt_rope: Optional[bool] = False):
        """
        :param dataset_dirs: dataset directories
        :param step_size: the number of time steps to skip when slicing the full trajectories for training
        """
        super(DynamicsDatasetLoader, self).__init__(dataset_dirs)

        self.use_gt_rope = use_gt_rope
        self.step_size = step_size
        self.scenario = get_scenario(self.hparams['scenario'])

        self.data_collection_params = self.hparams['data_collection_params']
        self.state_keys = self.data_collection_params['state_keys']
        self.state_keys.append('time_idx')

        self.action_keys = self.data_collection_params['action_keys']

        self.constant_feature_names = [
            'env',
            'extent',
            'origin',
            'res',
            'traj_idx',
        ]

        self.time_indexed_keys = self.state_keys + self.action_keys

        self.int64_keys = ['time_idx']

        if 'new_sequence_length' in self.hparams:
            self.steps_per_traj = self.hparams['new_sequence_length']
        else:
            self.steps_per_traj = self.data_collection_params['steps_per_traj']
        self.batch_metadata = {
            'sequence_length': self.steps_per_traj
        }

    def make_features_description(self):
        features_description = super().make_features_description()
        for feature_name in self.constant_feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        for feature_name in self.state_keys:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)
        for feature_name in self.action_keys:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        return features_description

    def split_into_sequences(self, example, desired_sequence_length):
        # return a dict where every element has different sequences split across the 0th dimension
        for start_t in range(0, self.steps_per_traj - desired_sequence_length + 1, desired_sequence_length):
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
        dataset = super().post_process(dataset, n_parallel_calls)

        if self.use_gt_rope:
            print(Fore.GREEN + "Using ground-truth rope state")
            dataset = dataset.map(use_gt_rope)

        return dataset

    def index_time_batched(self, example_batched, t: int):
        e_t = numpify(remove_batch(index_time_batched(example_batched, self.time_indexed_keys, t)))
        return e_t


def get_state_like_keys(params: Dict):
    all_keys = params['state_keys'] + params['observation_keys'] + params['observation_feature_keys']
    # make sure there are no duplicates
    all_keys = list(set(all_keys))
    return all_keys
