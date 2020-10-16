from typing import Optional, Dict

import numpy as np
import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts, numpify
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class ShootingMethod:

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: Optional[BaseConstraintChecker],
                 scenario: ExperimentScenario,
                 params: Dict,
                 verbose: Optional[int] = 0,
                 ):
        self.include_true_action = True
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.verbose = verbose
        self.scenario = scenario
        self.n_samples = params["n_samples"]
        self.rng = np.random.RandomState(0)

    def optimize(self,
                 current_observation: Dict,
                 environment: Dict,
                 goal_state: Dict,
                 start_state: Dict,
                 true_action: Dict,
                 ):

        # TODO: use scenario to sample
        current_left_gripper_position = current_observation['left_gripper']
        current_right_gripper_position = current_observation['right_gripper']
        m = 0.5
        if self.include_true_action:
            left_noise = self.rng.uniform(low=[-m] * 3, high=[m] * 3, size=[self.n_samples - 1, 1, 3])
            right_noise = self.rng.uniform(low=[-m] * 3, high=[m] * 3, size=[self.n_samples - 1, 1, 3])
            random_actions = {
                'left_gripper_position': current_left_gripper_position + left_noise,
                'right_gripper_position': current_right_gripper_position + right_noise,
            }
            for k, v in random_actions.items():
                random_actions[k] = tf.concat([random_actions[k], [[true_action[k]]]], axis=0)
        else:
            left_noise = self.rng.uniform(low=[-m] * 3, high=[m] * 3, size=[self.n_samples, 1, 3])
            right_noise = self.rng.uniform(low=[-m] * 3, high=[m] * 3, size=[self.n_samples, 1, 3])
            random_actions = {
                'left_gripper_position': current_left_gripper_position + left_noise,
                'right_gripper_position': current_right_gripper_position + right_noise,
            }

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
        print(costs)
        # print('min idx', min_idx.numpy())

        # cmap = cm.Blues
        # for i in range(self.n_samples):
        #     s = numpify({k: v[i] for k, v in start_state_batched.items()})
        #     a = numpify({k: v[i][0] for k, v in random_actions.items()})
        #     print(costs[i])
        #     c = np.exp(-0.001 * costs[i] ** 2)
        #     self.scenario.plot_action_rviz(s, a, label='samples', color=cmap(c), idx1=2 * i, idx2=2 * i + 1)

        best_actions = {k: v[min_idx] for k, v in random_actions.items()}
        best_prediction = {k: v[min_idx] for k, v in mean_predictions.items()}

        best_actions = numpify(dict_of_sequences_to_sequence_of_dicts(best_actions))
        best_predictions = numpify(dict_of_sequences_to_sequence_of_dicts(best_prediction))
        return best_actions, best_predictions
