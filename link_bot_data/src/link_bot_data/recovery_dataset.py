import pathlib
from typing import List

import tensorflow as tf

from link_bot_data.base_dataset import BaseDataset


class RecoveryDataset(BaseDataset):

    def __init__(self, dataset_dirs: List[pathlib.Path], load_true_states=False, no_balance=True):
        super(RecoveryDataset, self).__init__(dataset_dirs)
        self.no_balance = no_balance

        self.state_keys = self.hparams['state_keys']

        self.feature_names = [
            'full_env/env',
            'full_env/origin',
            'full_env/extent',
            'full_env/res',
            'traj_idx',
            'start_t',
            'end_t',
            'action',
            'mask',
        ]

        for k in self.hparams['states_description'].keys():
            self.feature_names.append(k)

    def make_features_description(self):
        features_description = {}
        for feature_name in self.feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        return features_description

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        def _rename_env(example):
            example['env'] = example['full_env/env']
            example['res'] = example['full_env/res']
            example['origin'] = example['full_env/origin']
            example['extent'] = example['full_env/extent']
            return example

        dataset = dataset.map(_rename_env)
        return dataset
