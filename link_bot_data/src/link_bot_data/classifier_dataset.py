import pathlib
from typing import List, Dict

import tensorflow as tf

from link_bot_data.base_dataset import BaseDataset
from link_bot_data.link_bot_dataset_utils import add_predicted
from link_bot_pycommon.get_scenario import get_scenario


class ClassifierDataset(BaseDataset):

    def __init__(self, dataset_dirs: List[pathlib.Path], load_true_states=False, no_balance=True):
        super(ClassifierDataset, self).__init__(dataset_dirs)
        self.no_balance = no_balance
        self.load_true_states = load_true_states
        self.labeling_params = self.hparams['labeling_params']
        self.label_state_key = self.hparams['labeling_params']['state_key']
        self.horizon = self.hparams['labeling_params']['classifier_horizon']
        scenario_params = {
            'scenario': self.hparams['scenario'],
            'data_collection_params': self.hparams['data_collection_params']
        }
        self.scenario = get_scenario(scenario_params)

        self.state_keys = self.hparams['state_keys']
        self.cache_negative = False

        self.feature_names = [
            'classifier_start_t',
            'classifier_end_t',
            'env',
            'extent',
            'origin',
            'res',
            'traj_idx',
            'prediction_start_t',
            'is_close',
        ]

        if self.load_true_states:
            for k in self.state_keys:
                self.feature_names.append(k)

        for k in self.state_keys:
            self.feature_names.append(add_predicted(k))

        for k in self.hparams['action_description'].keys():
            self.feature_names.append(k)

        self.feature_names.append(add_predicted('stdev'))

    def make_features_description(self):
        features_description = {}
        for feature_name in self.feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        return features_description

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        def _add_time(example: Dict):
            # this function is called before batching occurs, so the first dimension should be time
            example['time'] = tf.cast(example[add_predicted(self.state_keys[0])].shape[0], tf.int64)
            return example

        dataset = dataset.map(_add_time)
        return dataset
