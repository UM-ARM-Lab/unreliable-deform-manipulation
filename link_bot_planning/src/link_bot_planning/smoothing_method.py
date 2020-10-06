from typing import Optional, Dict, List
import numpy as np
import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts, numpify
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction
from state_space_dynamics.base_filter_function import BaseFilterFunction


class ShootingMethod:

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: Optional[BaseConstraintChecker],
                 filter_model: BaseFilterFunction,
                 scenario: ExperimentScenario,
                 params: Dict,
                 verbose: Optional[int] = 0,
                 ):
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.filter_model = filter_model
        self.verbose = verbose
        self.scenario = scenario
        self.n_samples = params["n_samples"]

    def optimize(self,
                 current_observation: Dict,
                 environment: Dict,
                 goal: Dict,
                 start_state: Dict,
                 ):
        rng = np.random.RandomState(0)

        current_left_gripper_position = current_observation['gripper1']
        current_right_gripper_position = current_observation['gripper2']
        random_actions = {
            'gripper1_position': current_left_gripper_position + rng.uniform(low=[-0.1] * 3, high=[0.1] * 3,
                                                                             size=[self.n_samples, 1, 3]),
            'gripper2_position': current_right_gripper_position + rng.uniform(low=[-0.1] * 3, high=[0.1] * 3,
                                                                              size=[self.n_samples, 1, 3]),
        }
        environment_batched = {k: tf.stack([v] * self.n_samples, axis=0) for k, v in environment.items()}
        start_state_batched = {k: tf.expand_dims(tf.stack([v] * self.n_samples, axis=0), axis=1) for k, v in start_state.items()}
        mean_predictions, _ = self.fwd_model.propagate_differentiable_batched(environment=environment_batched,
                                                                              state=start_state_batched,
                                                                              actions=random_actions)
        final_states = {k: v[:, -1] for k, v in mean_predictions.items()}
        goal_state, _ = self.filter_model.filter(environment, None, goal)
        goal_state_batched = {k: tf.stack([v] * self.n_samples, axis=0) for k, v in goal_state.items()}
        costs = self.scenario.trajopt_distance_to_goal_differentiable(final_states, goal_state_batched)
        min_idx = tf.math.argmin(costs, axis=0)

        best_actions = {k: v[min_idx] for k, v in random_actions.items()}
        best_prediction = {k: v[min_idx] for k, v in mean_predictions.items()}

        best_actions = numpify(dict_of_sequences_to_sequence_of_dicts(best_actions))
        best_predictions = numpify(dict_of_sequences_to_sequence_of_dicts(best_prediction))
        return best_actions, best_predictions
