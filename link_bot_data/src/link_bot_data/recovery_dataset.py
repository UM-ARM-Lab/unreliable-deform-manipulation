import pathlib
from typing import List

import tensorflow as tf

import rospy
from link_bot_data.base_dataset import BaseDatasetLoader
from link_bot_data.dataset_utils import filter_and_cache
from link_bot_data.visualization import init_viz_env, recovery_probability_viz_t, init_viz_action
from merrrt_visualization.rviz_animation_controller import RvizAnimation
from std_msgs.msg import Float32


class RecoveryDatasetLoader(BaseDatasetLoader):

    def __init__(self, dataset_dirs: List[pathlib.Path]):
        super(RecoveryDatasetLoader, self).__init__(dataset_dirs)
        self.state_keys = self.hparams['state_keys']
        self.action_keys = self.hparams['action_keys']

        self.feature_names = [
            'env',
            'origin',
            'extent',
            'res',
            'traj_idx',
            'start_t',
            'end_t',
            'accept_probabilities'
        ]

        for k in self.hparams['state_keys']:
            self.feature_names.append(k)

        self.horizon = self.hparams["labeling_params"]["action_sequence_horizon"]
        self.n_action_samples = self.hparams["labeling_params"]["n_action_samples"]

        self.predicted_state_keys = self.hparams['predicted_state_keys']
        self.true_state_keys = self.hparams['true_state_keys']

        for k in self.state_keys:
            self.feature_names.append(k)

        for k in self.action_keys:
            self.feature_names.append(k)

        self.batch_metadata = {
            'time': 2
        }

    def make_features_description(self):
        features_description = {}
        for feature_name in self.feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        return features_description

    @staticmethod
    def is_stuck(example):
        stuck = example['recovery_probability'][0] <= 0
        return stuck

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        def _add_recovery_probabilities(example):
            n_accepts = tf.math.count_nonzero(example['accept_probabilities'] > 0.5, axis=1)
            example['recovery_probability'] = tf.cast(n_accepts / self.n_action_samples, tf.float32)
            return example

        dataset = dataset.map(_add_recovery_probabilities)
        dataset = filter_and_cache(dataset, RecoveryDatasetLoader.is_stuck)
        return dataset

    def anim_rviz(self, example):
        recovery_probability_pub = rospy.Publisher("stdev", Float32, queue_size=10)
        anim = RvizAnimation(scenario=self.scenario,
                             n_time_steps=self.horizon,
                             init_funcs=[init_viz_env,
                                         self.init_viz_action(),
                                         recovery_probability_viz_t(recovery_probability_pub),
                                         ],
                             t_funcs=[init_viz_env, self.classifier_transition_viz_t()])
        anim.play(example)

        # recovery_probability = example['recovery_probability'][1]
        # color_factor = log_scale_0_to_1(recovery_probability, k=10)
        # s_0 = {k: example[k][0] for k in state_keys}
        # s_1 = {k: example[k][1] for k in state_keys}
        # a = {k: example[k][0] for k in action_keys}
        #
        #     scenario.plot_action_rviz(s_0, a, label='observed')
        #     scenario.plot_state_rviz(s_0, label='observed', idx=1, color='w')
        #     scenario.plot_state_rviz(s_1, label='observed', idx=2, color=cm.Reds(color_factor))
        #     sleep(0.01)

    def init_viz_action(self):
        return init_viz_action(self.scenario_metadata, self.action_keys, self.predicted_state_keys)
