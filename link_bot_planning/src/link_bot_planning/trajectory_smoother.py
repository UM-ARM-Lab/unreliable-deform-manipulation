from time import perf_counter
from typing import Dict

import numpy as np
import tensorflow as tf
from more_itertools import pairwise

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_planning.planning_scenario import PlanningScenario
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class TrajectorySmoother:

    def __init__(self,
                 verbose: int,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: BaseConstraintChecker,
                 planning_scenario: PlanningScenario,
                 params: Dict):
        self.verbose = verbose
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.iters = params["iters"]
        self.goal_alpha = params["goal_alpha"]
        self.constraints_alpha = params["constraints_alpha"]
        self.action_alpha = params["action_alpha"]
        self.planning_scenario = planning_scenario
        self.optimizer = tf.keras.optimizers.Adam(0.02)

    def smooth(self,
               full_env: np.ndarray,
               full_env_origin: np.ndarray,
               res: float,
               goal,
               actions: np.ndarray,
               planned_path: Dict,
               ):
        actions = tf.Variable(actions, dtype=tf.float32, name='controls', trainable=True)

        start_smoothing_time = perf_counter()
        for i in range(self.iters):
            actions, planned_path, _ = self.step(full_env, full_env_origin, res, goal, actions, planned_path)
        smoothing_time = perf_counter() - start_smoothing_time

        if self.verbose >= 1:
            print("Smoothing time: {:.3f}".format(smoothing_time))

        return actions, planned_path

    def step(self, full_env, full_env_origin, res, goal, actions, planned_path):
        start_states = planned_path[0]
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            # Compute the states predicted given the actions
            predictions = self.fwd_model.propagate_differentiable(full_env=full_env,
                                                                  full_env_origin=full_env_origin,
                                                                  res=res,
                                                                  start_states=start_states,
                                                                  actions=actions)

            # Compute the constraint values predicted given the states and actions
            constraint_loss = 0
            for t in range(len(predictions)):
                predictions_t = predictions[:t]
                constraint_prediction_t = self.classifier_model.check_constraint_differentiable(full_env=full_env,
                                                                                                full_env_origin=full_env_origin,
                                                                                                res=res,
                                                                                                states_trajs=predictions_t,
                                                                                                actions=actions)
                # TODO: try setting this to be the max constraint_prediction_t only
                constraint_loss += constraint_prediction_t

            # Compute various loss terms
            final_state = predictions[-1]
            goal_loss = self.planning_scenario.distance_to_goal_differentiable(final_state, goal)

            distances = [self.planning_scenario.distance_differentiable(s1, s2) for (s1, s2) in pairwise(predictions)]
            length_loss = tf.reduce_sum(tf.square(distances))

            action_loss = tf.reduce_sum(tf.square(actions))
            losses = [length_loss, goal_loss, constraint_loss, action_loss]
            weights = tf.convert_to_tensor([1, self.goal_alpha, self.constraints_alpha, self.action_alpha], dtype=tf.float32)
            weighted_losses = tf.multiply(losses, weights)
            loss = tf.reduce_sum(weighted_losses, axis=0)
        variables = [actions]
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return actions, predictions, [weighted_losses[0], weighted_losses[1], weighted_losses[2], weighted_losses[3], loss]
