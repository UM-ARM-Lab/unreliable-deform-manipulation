import pathlib
from typing import List

import tensorflow as tf

from link_bot_data.base_dataset import BaseDataset
from link_bot_data.link_bot_dataset_utils import split_into_sequences, slice_sequences
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.params import FullEnvParams


class DynamicsDataset(BaseDataset):
    def __init__(self, dataset_dirs: List[pathlib.Path], step_size: int = 1):
        """
        :param dataset_dirs: dataset directories
        :param step_size: the number of time steps to skip when slicing the full trajectories into trajectories for training
        """
        super(DynamicsDataset, self).__init__(dataset_dirs)

        self.step_size = step_size
        self.full_env_params = FullEnvParams.from_json(self.hparams['full_env_params'])
        self.scenario = get_scenario(self.hparams['scenario'])

        self.action_feature_names = ['action']

        self.state_feature_names = [
            'time_idx',
            'traj_idx',
        ]

        self.states_description = self.hparams['states_description']
        for state_key in self.states_description.keys():
            self.state_feature_names.append('{}'.format(state_key))

        self.constant_feature_names = [
            'full_env/env',
            'full_env/extent',
            'full_env/origin',
            'full_env/res',
        ]

    def make_features_description(self):
        features_description = {}
        for feature_name in self.constant_feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        for i in range(self.max_sequence_length):
            for feature_name in self.state_feature_names:
                feature_name = "%d/" + feature_name
                features_description[feature_name % i] = tf.io.FixedLenFeature([], tf.string)
        for i in range(self.max_sequence_length - 1):
            for feature_name in self.action_feature_names:
                feature_name = "%d/" + feature_name
                features_description[feature_name % i] = tf.io.FixedLenFeature([], tf.string)

        return features_description

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):

        def _split_into_sequences(example):
            return split_into_sequences(self.state_feature_names,
                                        self.action_feature_names,
                                        self.constant_feature_names,
                                        max_sequence_length=self.max_sequence_length,
                                        example_dict=example)

        def _slice_sequences(constant_data, state_like_seqs, action_like_seqs):
            return slice_sequences(constant_data, state_like_seqs, action_like_seqs, self.desired_sequence_length, self.step_size)

        # FIXME: don't separate const/state/action to begin with?
        def _combine_data(const_data, state_like_sequences, action_like_sequences):
            input_dict = {}
            for k, v in state_like_sequences.items():
                # chop off the last time step since that's not part of the input
                input_dict[k] = v[:-1]
            input_dict.update(action_like_sequences)
            input_dict.update(const_data)
            output_dict = {}
            output_dict.update(state_like_sequences)
            return input_dict, output_dict

        dataset = dataset.map(_split_into_sequences, num_parallel_calls=n_parallel_calls)
        dataset = dataset.flat_map(_slice_sequences)
        dataset = dataset.map(_combine_data, num_parallel_calls=n_parallel_calls)
        return dataset
