from typing import Optional, Dict
import numpy as np

import tensorflow as tf
from matplotlib import cm

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts, numpify, sequence_of_dicts_to_dict_of_sequences, \
    sequence_of_dicts_to_dict_of_tensors
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class ShootingMethod:

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: Optional[BaseConstraintChecker],
                 scenario: ExperimentScenario,
                 params: Dict,
                 verbose: Optional[int] = 0,
                 ):
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.verbose = verbose
        self.scenario = scenario
        self.n_samples = params["n_samples"]
        self.action_rng = np.random.RandomState(0)

    def optimize(self,
                 current_observation: Dict,
                 environment: Dict,
                 goal_state: Dict,
                 start_state: Dict,
                 ):
        random_actions = self.scenario.sample_action_sequences(action_rng=self.action_rng,
                                                               action_sequence_length=1,
                                                               n_action_sequences=self.n_samples,
                                                               environment=environment,
                                                               state=start_state,
                                                               data_collection_params=self.fwd_model.data_collection_params,
                                                               action_params=self.fwd_model.data_collection_params)
        random_actions = [sequence_of_dicts_to_dict_of_tensors(a) for a in random_actions]
        random_actions = sequence_of_dicts_to_dict_of_tensors(random_actions)

        environment_batched = {k: tf.stack([v] * self.n_samples, axis=0) for k, v in environment.items()}
        start_state_batched = {k: tf.expand_dims(tf.stack([v] * self.n_samples, axis=0), axis=1) for k, v in start_state.items()}
        mean_predictions, _ = self.fwd_model.propagate_differentiable_batched(environment=environment_batched,
                                                                              state=start_state_batched,
                                                                              actions=random_actions)

        final_states = {k: v[:, -1] for k, v in mean_predictions.items()}
        goal_state_batched = {k: tf.stack([v] * self.n_samples, axis=0) for k, v in goal_state.items()}
        costs = self.scenario.trajopt_distance_to_goal_differentiable(final_states, goal_state_batched)
        costs = tf.squeeze(costs)
        min_idx = tf.math.argmin(costs, axis=0)
        # print(costs)
        # print('min idx', min_idx.numpy())
        best_indices = tf.argsort(costs)

        cmap = cm.Blues
        n_to_show = 5
        min_cost = costs[best_indices[0]]
        max_cost = costs[best_indices[n_to_show]]
        for j, i in enumerate(best_indices[:n_to_show]):
            s = numpify({k: v[i] for k, v in start_state_batched.items()})
            a = numpify({k: v[i][0] for k, v in random_actions.items()})
            c = (costs[i] - min_cost) / (max_cost - min_cost)
            self.scenario.plot_action_rviz(s, a, label='samples', color=cmap(c), idx1=2 * j, idx2=2 * j + 1)

        best_actions = {k: v[min_idx] for k, v in random_actions.items()}
        best_prediction = {k: v[min_idx] for k, v in mean_predictions.items()}

        best_actions = numpify(dict_of_sequences_to_sequence_of_dicts(best_actions))
        best_predictions = numpify(dict_of_sequences_to_sequence_of_dicts(best_prediction))
        return best_actions, best_predictions
