from typing import Dict

import numpy as np
import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.link_bot_pycommon import print_dict
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class TrajectorySmoother:

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: BaseConstraintChecker,
                 goal_point_idx: int,
                 goal_subspace_name: str,
                 params: Dict):
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.iters = params["iters"]
        self.goal_alpha = params["goal_alpha"]
        self.constraints_alpha = params["constraints_alpha"]
        self.action_alpha = params["action_alpha"]
        self.goal_point_idx = goal_point_idx
        self.goal_subspace_feature = 'state/{}'.format(goal_subspace_name)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def smooth(self,
               full_env: np.ndarray,
               full_env_origin: np.ndarray,
               res: float,
               goal_point: np.ndarray,
               actions: np.ndarray,
               planned_path: Dict[str, np.ndarray],
               ):
        actions = tf.Variable(actions, dtype=tf.float32, name='controls', trainable=True)

        for i in range(self.iters):
            actions = self.step(full_env, full_env_origin, res, goal_point, actions, planned_path)[0]
        return actions, planned_path

    def step(self, full_env, full_env_origin, res, goal_point, actions, planned_path):
        start_states = dict([(k, v[0]) for k, v in planned_path.items()])
        T = actions.shape[0]
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            # Compute the states predicted given the actions
            predictions = self.fwd_model.propagate_differentiable(full_env=full_env,
                                                                  full_env_origin=full_env_origin,
                                                                  res=res,
                                                                  start_states=start_states,
                                                                  actions=actions)

            # Compute the constraint values predicted given the states and actions
            constraint_predictions = self.classifier_model.check_constraint_differentiable(full_env=full_env,
                                                                                           full_env_origin=full_env_origin,
                                                                                           res=res,
                                                                                           states_trajs=predictions,
                                                                                           actions=actions)

            goal_subspace_predictions = predictions[self.goal_subspace_feature]
            predicted_points = tf.reshape(goal_subspace_predictions, [T + 1, -1, 2])
            deltas = predicted_points[1:] - predicted_points[:-1]
            final_target_point_pred = predicted_points[-1, self.goal_point_idx]
            goal_loss = tf.reduce_sum(tf.square(final_target_point_pred - goal_point))
            length_loss = tf.reduce_sum(tf.square(deltas))
            constraint_loss = tf.reduce_sum(constraint_predictions)
            action_loss = tf.reduce_sum(tf.square(actions))
            losses = [length_loss, goal_loss, constraint_loss, action_loss]
            weights = tf.convert_to_tensor([1, self.goal_alpha, self.constraints_alpha, self.action_alpha], dtype=tf.float32)
            weighted_losses = tf.multiply(losses, weights)
            loss = tf.reduce_sum(weighted_losses, axis=0)
        variables = [actions]
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return actions, predictions, weighted_losses[0], weighted_losses[1], weighted_losses[2], weighted_losses[3], loss
