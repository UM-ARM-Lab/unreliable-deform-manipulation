from typing import Dict
import tensorflow as tf
import json
import numpy as np
import pathlib
import tensorflow_probability as tfp

from matplotlib import cm
from moonshine.moonshine_utils import gather_dict, add_batch, remove_batch, index_dict_of_batched_vectors_tf
from link_bot_pycommon.pycommon import log_scale_0_to_1, make_dict_tf_float32
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import numpify, dict_of_sequences_to_sequence_of_dicts_tf, sequence_of_dicts_to_dict_of_tensors
from link_bot_classifiers.base_recovery_policy import BaseRecoveryPolicy
from link_bot_data.recovery_dataset import RecoveryDataset


class KNNRecoveryPolicy(BaseRecoveryPolicy):

    def __init__(self, hparams: Dict, model_dir: pathlib.Path, scenario: ExperimentScenario, rng: np.random.RandomState):
        super().__init__(hparams, model_dir, scenario, rng)
        self.model_dir = model_dir
        self.scenario = scenario
        # models are data and data are models
        self.dataset = RecoveryDataset([model_dir])
        self.noise_rng = np.random.RandomState(0)
        self.sample_seed = 0

        def _nonzero_prob(example):
            return example['recovery_probability'][1] > 1e-2

        self.tf_dataset = self.dataset.get_datasets(mode='train')
        self.tf_dataset = self.tf_dataset.filter(_nonzero_prob)
        self.cached_dataset = []
        for example in self.tf_dataset:
            self.cached_dataset.append(example)
        self.cached_dataset = sequence_of_dicts_to_dict_of_tensors(self.cached_dataset)

    def __call__(self, environment: Dict, state: Dict):
        # indexing by 0 just gets rid of the time dimension, whould shuld always be 1 here
        dataset_states = {k: self.cached_dataset[k][:, 0] for k in self.dataset.state_keys}
        dataset_actions = {k: self.cached_dataset[k][:, 0] for k in self.dataset.action_keys}
        dataset_states_local = self.scenario.put_state_local_frame(dataset_states)
        state_local = self.scenario.put_state_local_frame(make_dict_tf_float32(state))
        distance = self.scenario.full_distance_tf(dataset_states_local, state_local)
        # ignore the recovery probability of t=0 because it is assumed to be 0, since we're calling the recovery policy
        recovery_probability = self.cached_dataset['recovery_probability'][:, 1]
        weight = tf.math.divide_no_nan(recovery_probability, distance)

        # Weights define the likelihoods of sampling
        dist = tfp.distributions.Categorical(logits=tf.math.log(weight))

        # Sample
        for i in range(10):
            sampled_index = dist.sample(seed=self.sample_seed)
            sampled_state = {k: self.cached_dataset[k][sampled_index, 0] for k in self.dataset.state_keys}
            sampled_action = {k: self.cached_dataset[k][sampled_index, 0] for k in self.dataset.action_keys}
            sampled_local_action = self.scenario.put_action_local_frame(sampled_state, sampled_action)
            action_applied_at_state = self.scenario.apply_local_action_at_state(state, sampled_local_action)
            action_applied_at_state_noisy = self.scenario.add_noise(action_applied_at_state, self.noise_rng)

            # visualize
            self.scenario.plot_state_rviz(sampled_state, label='sample', color='gray', idx=i)
            self.scenario.plot_action_rviz(sampled_state, sampled_action, label='sample', color='m', idx=i)
            self.scenario.plot_action_rviz(state, action_applied_at_state_noisy, label='proposed', color='m', idx=i)

        return action_applied_at_state_noisy
