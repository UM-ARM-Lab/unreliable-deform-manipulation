from typing import Dict

import numpy as np
import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class TrajectorySmoother:

    def __init__(self,
                 fwd_model: BaseDynamicsFunction,
                 classifier_model: BaseConstraintChecker,
                 goal_point_idx: int,
                 params: Dict):
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.iters = params["iters"]
        self.goal_alpha = params["goal_alpha"]
        self.constraints_alpha = params["constraints_alpha"]
        self.action_alpha = params["action_alpha"]
        self.goal_point_idx = goal_point_idx

    def smooth(self,
               goal_point: np.ndarray,
               controls: np.ndarray,
               planned_path: Dict[str, np.ndarray],
               ):
        controls = tf.Variable(controls, dtype=tf.float32, name='controls', trainable=True)
        for i in range(self.iters):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                tape.watch(state)
                # input_dict = {
                #     fwd_model.net.state_feature: state,
                #     'action': controls,
                # }
                predictions = self.fwd_model.propagate_differentiable()
                predicted_points = tf.reshape(predictions[self.fwd_model.state_keys], [T + 1, -1, 2])
                deltas = predicted_points[1:] - predicted_points[:-1]
                final_target_point_pred = predicted_points[-1, self.goal_point_idx]
                goal_loss = tf.linalg.norm(final_target_point_pred - goal_point)
                distances = tf.linalg.norm(deltas)
                length_loss = tf.reduce_sum(distances)
                constraint_loss = 0  # TODO:
                loss = length_loss + self.goal_alpha * goal_loss + self.constraints_alpha * constraint_loss

            variables = [controls]
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            path_lengths.append(length_loss)
            paths.append(predicted_points)
            final_points.append(final_target_point_pred)
        return controls, planned_path
