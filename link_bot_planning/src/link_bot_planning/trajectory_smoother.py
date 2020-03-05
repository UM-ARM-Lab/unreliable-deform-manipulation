from time import perf_counter
from typing import Dict, List

import numpy as np
import tensorflow as tf
from more_itertools import pairwise

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_planning.experiment_scenario import ExperimentScenario
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class TrajectorySmoother:

    def __init__(self,
                 verbose: int,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: BaseConstraintChecker,
                 experiment_scenario: ExperimentScenario,
                 params: Dict):
        self.verbose = verbose
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.iters = params["iters"]
        self.length_alpha = params["length_alpha"]
        self.goal_alpha = params["goal_alpha"]
        self.constraints_alpha = params["constraints_alpha"]
        self.action_alpha = params["action_alpha"]
        self.experiment_scenario = experiment_scenario
        self.optimizer = tf.keras.optimizers.Adam(params["initial_learning_rate"], amsgrad=True)

    def smooth(self,
               full_env: np.ndarray,
               full_env_origin: np.ndarray,
               res: float,
               goal,
               actions: np.ndarray,
               planned_path: List[Dict],
               ):
        actions = tf.Variable(actions, dtype=tf.float32, name='controls', trainable=True)

        start_smoothing_time = perf_counter()
        for i in range(self.iters):
            actions, planned_path, _, _ = self.step(full_env, full_env_origin, res, goal, actions, planned_path)
        smoothing_time = perf_counter() - start_smoothing_time

        if self.verbose >= 1:
            print("Smoothing time: {:.3f}".format(smoothing_time))

        return actions, planned_path

    def step(self, full_env, full_env_origin, res, goal, actions, planned_path):
        start_states = planned_path[0]
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            # Compute the states predicted given the actions
            dynamics_t0 = perf_counter()
            predictions = self.fwd_model.propagate_differentiable(full_env=full_env,
                                                                  full_env_origin=full_env_origin,
                                                                  res=res,
                                                                  start_states=start_states,
                                                                  actions=actions)
            dynamics_dt = perf_counter() - dynamics_t0
            if self.verbose >= 1:
                print("Dynamics Fwd Pass Took {:.3f}s".format(dynamics_dt))

            # Compute the constraint values predicted given the states and actions
            constraint_t0 = perf_counter()
            constraint_costs = []
            for t in range(1, len(predictions)):
                predictions_t = predictions[:t + 1]
                constraint_prediction_t = self.classifier_model.check_constraint_differentiable(full_env=full_env,
                                                                                                full_env_origin=full_env_origin,
                                                                                                res=res,
                                                                                                states_sequence=predictions_t,
                                                                                                actions=actions)
                print(constraint_prediction_t)
                # NOTE: this math maps (0 -> inf, 1->0)
                constraint_cost = tf.square(1.0 - constraint_prediction_t)
                constraint_costs.append(constraint_cost)
            constraint_loss = tf.reduce_sum(constraint_costs)
            constraint_dt = perf_counter() - constraint_t0
            if self.verbose >= 1:
                print("ConstraintsFwd Pass Took {:.3f}s".format(constraint_dt))

            # Compute various loss terms
            final_state = predictions[-1]
            goal_loss = self.experiment_scenario.distance_to_goal_differentiable(final_state, goal)

            distances = [self.experiment_scenario.distance_differentiable(s1, s2) for (s1, s2) in pairwise(predictions)]
            length_loss = tf.reduce_sum(tf.square(distances))

            action_loss = tf.reduce_sum(tf.reduce_sum(tf.square(actions), axis=1))
            losses = [length_loss, goal_loss, constraint_loss, action_loss]
            weights = tf.convert_to_tensor([self.length_alpha, self.goal_alpha, self.constraints_alpha, self.action_alpha],
                                           dtype=tf.float32)
            weighted_losses = tf.multiply(losses, weights)
            loss = tf.reduce_sum(weighted_losses, axis=0)

        variables = [actions]
        t0 = perf_counter()
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        time_to_update = perf_counter() - t0

        if self.verbose >= 1:
            print("Gradient Apply Took {:.3f}s".format(time_to_update))

        return actions, predictions, [weighted_losses[0], weighted_losses[1], weighted_losses[2], weighted_losses[3],
                                      loss], length_loss
