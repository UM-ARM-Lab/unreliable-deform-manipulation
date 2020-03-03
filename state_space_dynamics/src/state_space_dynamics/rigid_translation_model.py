import pathlib
from typing import Dict

import numpy as np
import tensorflow as tf

from link_bot_pycommon.link_bot_pycommon import n_state_to_n_points
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class RigidTranslationModel(BaseDynamicsFunction):

    def __init__(self, model_dir: pathlib.Path, batch_size: int):
        super().__init__(model_dir, batch_size)
        self.beta = self.hparams['beta']
        self.batch_size = batch_size
        self.B = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32) * self.beta

    def propagate_differentiable(self,
                                 full_env: np.ndarray,
                                 full_env_origin: np.ndarray,
                                 res: float,
                                 start_states: Dict[str, np.ndarray],
                                 actions: tf.Variable) -> Dict[str, tf.Tensor]:
        """
        :param full_env:        (H, W)
        :param full_env_origin: (2)
        :param res:             scalar
        :param start_states:          each value in the dictionary should be of shape (batch, n_state)
        :param actions:        (T, 2)
        :return: states:       each value in the dictionary should be a of shape [batch, T+1, n_state)
        """
        predictions = {}
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        for state_feature, start_state in start_states.items():
            s_t = tf.convert_to_tensor(start_state, dtype=tf.float32)
            n_points = n_state_to_n_points(s_t.shape[0])
            pred_states = [s_t]
            for t in range(actions.shape[0]):
                action_t = actions[t]
                delta_s_t = tf.tensordot(action_t, self.B, axes=1)
                delta_s_t_flat = tf.tile(delta_s_t, [n_points])
                s_t = s_t + delta_s_t_flat * self.dt
                pred_states.append(s_t)

            pred_states = tf.stack(pred_states, axis=0)
            predictions[state_feature] = pred_states
        return predictions
