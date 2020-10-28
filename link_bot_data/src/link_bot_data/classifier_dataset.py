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
        self.scenario = get_scenario(self.hparams['scenario'])

        self.true_state_keys = self.hparams['true_state_keys']
        self.predicted_state_keys = self.hparams['predicted_state_keys']
        self.action_keys = self.hparams['action_keys']

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

        scenario_metadata = self.hparams['scenario_metadata']

        self.batch_metadata = {
            'time': self.horizon
        }

        if self.load_true_states:
            for k in self.true_state_keys:
                self.feature_names.append(k)

        for k in self.predicted_state_keys:
            self.feature_names.append(add_predicted(k))

        for k in self.action_keys:
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
            example['time'] = tf.cast(self.horizon, tf.int64)
            return example

        # dataset = dataset.map(_add_time)

        def _add_rope_noise(example):
            example[add_predicted('link_bot')] = example[add_predicted('link_bot')] + tf.random.normal([75], 0, 0.01)
            return example

        # dataset = dataset.map(_add_rope_noise)

        # this is used for adding joint_names
        scenario_metadata = self.hparams['scenario_metadata']

        def _add_scenario_metadata(example: Dict):
            example.update(scenario_metadata)
            return example

        dataset = dataset.map(_add_scenario_metadata)

        return dataset
