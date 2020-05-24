import pathlib
from typing import List

import tensorflow as tf

from link_bot_data.base_dataset import BaseDataset
from link_bot_data.link_bot_dataset_utils import add_planned, balance
from link_bot_pycommon.params import FullEnvParams


class ClassifierDataset(BaseDataset):

    def __init__(self, dataset_dirs: List[pathlib.Path], load_true_states=False, no_balance=True):
        super(ClassifierDataset, self).__init__(dataset_dirs)
        self.no_balance = no_balance
        self.load_true_states = load_true_states
        self.full_env_params = FullEnvParams.from_json(self.hparams['full_env_params'])
        self.labeling_params = self.hparams['labeling_params']
        self.label_state_key = self.hparams['labeling_params']['state_key']

        self.state_keys = self.hparams['state_keys']
        self.cache_negative = False

        self.feature_names = [
            'full_env/env',
            'full_env/origin',
            'full_env/extent',
            'full_env/res',
            'traj_idx',
            'prediction_start_t',
            'classifier_start_t',
            'classifier_end_t',
            'is_close',
            'action',
        ]

        if self.load_true_states:
            for k in self.hparams['states_description'].keys():
                self.feature_names.append(k)

        for k in self.state_keys:
            self.feature_names.append(add_planned(k))

        self.feature_names.append(add_planned('stdev'))

    def make_features_description(self):
        features_description = {}
        for feature_name in self.feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        return features_description

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        if not self.no_balance:
            dataset = balance(dataset, self.labeling_params, cache_negative=self.cache_negative)

        return dataset
