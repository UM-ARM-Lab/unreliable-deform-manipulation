import pathlib
from typing import List, Dict, Optional

import tensorflow as tf
from merrrt_visualization.rviz_animation_controller import RvizAnimationController

import rospy
from link_bot_data.base_dataset import BaseDataset
from link_bot_data.link_bot_dataset_utils import add_predicted
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.moonshine_utils import numpify


class ClassifierDataset(BaseDataset):

    def __init__(self,
                 dataset_dirs: List[pathlib.Path],
                 load_true_states=False,
                 no_balance=True,
                 threshold: Optional[float] = None):
        super(ClassifierDataset, self).__init__(dataset_dirs)
        self.no_balance = no_balance
        self.load_true_states = load_true_states
        self.labeling_params = self.hparams['labeling_params']
        self.threshold = threshold if threshold is not None else self.labeling_params['threshold']
        rospy.loginfo(f"classifier using threshold {self.threshold}")
        self.horizon = self.hparams['labeling_params']['classifier_horizon']
        self.scenario = get_scenario(self.hparams['scenario'])

        self.true_state_keys = self.hparams['true_state_keys']
        self.true_state_keys.append('error')
        self.scenario_metadata = self.hparams['scenario_metadata']
        self.predicted_state_keys = [add_predicted(k) for k in self.hparams['predicted_state_keys']]
        self.predicted_state_keys.append(add_predicted('stdev'))
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

        self.batch_metadata = {
            'time': self.horizon
        }

        if self.load_true_states:
            for k in self.true_state_keys:
                self.feature_names.append(k)

        for k in self.predicted_state_keys:
            self.feature_names.append(k)

        for k in self.action_keys:
            self.feature_names.append(k)

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

        # this is used for adding joint_names
        scenario_metadata = self.scenario_metadata

        def _add_scenario_metadata(example: Dict):
            example.update(scenario_metadata)
            return example

        dataset = dataset.map(_add_scenario_metadata)

        threshold = self.threshold

        def _label(example: Dict):
            is_close = example['error'] < threshold
            example['is_close'] = tf.cast(is_close, dtype=tf.float32)
            return example

        dataset = dataset.map(_label)

        return dataset

    def plot_transition_rviz(self, example: Dict):
        example = numpify(example)
        self.scenario.plot_environment_rviz(example)

        anim = RvizAnimationController(n_time_steps=2)
        action = {k: example[k][0] for k in self.action_keys}
        pred_0 = self.index_pred_state_time(example, 0)
        self.scenario.plot_action_rviz(pred_0, action)
        while not anim.done:
            t = anim.t()

            pred_t = self.index_pred_state_time(example, t)
            self.scenario.plot_state_rviz(pred_t, label='predicted', color='#0000ffff')

            label_t = example['is_close'][1]
            self.scenario.plot_is_close(label_t)

            # true state(not known to classifier!)
            true_t = self.index_true_state_time(example, t)
            self.scenario.plot_state_rviz(true_t, label='actual', color='#ff0000ff', scale=1.1)
            anim.step()

    def index_true_state_time(self, example: Dict, t: int):
        e_t = {}
        for k in self.true_state_keys:
            e_t[k] = example[k][t]
        e_t.update(self.scenario_metadata)

        return e_t

    def index_pred_state_time(self, example: Dict, t: int):
        e_t = {}
        for k in self.predicted_state_keys:
            e_t[k] = example[k][t]
        e_t.update(self.scenario_metadata)

        return e_t
