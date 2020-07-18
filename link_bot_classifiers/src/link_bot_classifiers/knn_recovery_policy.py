from typing import Dict
import tensorflow as tf
import json
import numpy as np
import pathlib

from matplotlib import cm
from moonshine.moonshine_utils import gather_dict, add_batch, remove_batch, index_dict_of_batched_vectors_tf
from link_bot_pycommon.pycommon import log_scale_0_to_1, make_dict_tf_float32
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import numpify, dict_of_sequences_to_sequence_of_dicts_tf
from link_bot_classifiers.base_recovery_policy import BaseRecoveryPolicy
from link_bot_data.recovery_dataset import RecoveryDataset


class KNNRecoveryPolicy(BaseRecoveryPolicy):

    def __init__(self, hparams: Dict, model_dir: pathlib.Path, scenario: ExperimentScenario, rng: np.random.RandomState):
        super().__init__(hparams, model_dir, scenario, rng)
        self.model_dir = model_dir
        self.scenario = scenario
        # models are data and data are models
        self.dataset = RecoveryDataset([model_dir])
        self.tf_dataset = self.dataset.get_datasets(mode='train').batch(512)
        self.noise_rng = np.random.RandomState(0)

    def __call__(self, environment: Dict, state: Dict):
        weighted_local_action = {}
        weight_normalizing_factor = 0
        for example in self.tf_dataset:
            # indexing by 0 just gets rid of the time dimension, whould shuld always be 1 here
            dataset_state = {k: example[k][:, 0] for k in self.dataset.state_keys}
            dataset_action = {k: example[k][:, 0] for k in self.dataset.action_keys}
            dataset_state_local = self.scenario.put_state_local_frame(dataset_state)
            state_local = self.scenario.put_state_local_frame(make_dict_tf_float32(state))
            distance = self.scenario.full_distance_tf(dataset_state_local, state_local)
            # ignore the recovery probability of t=0 because it is assumed to be 0, since we're calling the recovery policy
            recovery_probability = example['recovery_probability'][:, 1]
            weight = tf.expand_dims(tf.math.divide_no_nan(recovery_probability, distance), axis=0)
            dataset_local_action = self.scenario.put_action_local_frame(dataset_state, dataset_action)
            for k in dataset_local_action.keys():
                if k not in weighted_local_action:
                    weighted_local_action[k] = tf.squeeze(tf.matmul(weight, dataset_local_action[k]), axis=0)
                weighted_local_action[k] += tf.squeeze(tf.matmul(weight, dataset_local_action[k]), axis=0)
            weight_normalizing_factor += tf.reduce_sum(weight)

        # all_nearby_examples = dict_of_sequences_to_sequence_of_dicts_tf(all_nearby_examples)
        # all_nearby_examples = sorted(all_nearby_examples, key=lambda e: e['recovery_probability'][0, 1], reverse=True)
        # for good_nearby_example in all_nearby_examples[:10]:
        #     good_nearby_action = {k: good_nearby_example[k][0, 0] for k in self.dataset.action_keys}
        #     good_nearby_state = {k: good_nearby_example[k][0, 0] for k in self.dataset.state_keys}
        #     recovery_probability = good_nearby_example['recovery_probability'][0, 1]
        #     local_action = self.scenario.put_action_local_frame(good_nearby_state, good_nearby_action)
        #     action_applied_at_state = self.scenario.apply_local_action_at_state(state, local_action)

        #     self.scenario.plot_recovery_probability(recovery_probability)
        #     color_factor = log_scale_0_to_1(tf.squeeze(recovery_probability), k=10)
        #     self.scenario.plot_state_rviz(good_nearby_state, label='nearby')
        #     self.scenario.plot_action_rviz(good_nearby_state, good_nearby_action,
        #                                    label='nearby', color=cm.Greens(color_factor))
        #     self.scenario.plot_action_rviz(state, action_applied_at_state,
        #                                    label='proposed', color=cm.Greens(color_factor))
        # self.scenario.plot_recovery_probability(recovery_probability)
        # color_factor = log_scale_0_to_1(tf.squeeze(recovery_probability), k=10)
        # self.scenario.plot_state_rviz(best_nearby_state, label='nearby')
        # self.scenario.plot_action_rviz(best_nearby_state, best_nearby_action,
        #                                label='nearby', color=cm.Greens(color_factor))

        weighted_local_action = {k: v/weight_normalizing_factor for k, v in weighted_local_action.items()}
        action_applied_at_state = self.scenario.apply_local_action_at_state(state, weighted_local_action)
        action_applied_at_state_noisy = self.scenario.add_noise(action_applied_at_state, self.noise_rng)
        self.scenario.plot_action_rviz(state, action_applied_at_state_noisy, label='proposed', color='magenta')
        return action_applied_at_state_noisy
