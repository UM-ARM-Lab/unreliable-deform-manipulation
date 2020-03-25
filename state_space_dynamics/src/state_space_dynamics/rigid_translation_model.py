import pathlib
from typing import Dict, List

import numpy as np
import tensorflow as tf

from link_bot_pycommon.link_bot_pycommon import n_state_to_n_points
from link_bot_planning.experiment_scenario import ExperimentScenario
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class RigidTranslationModel(BaseDynamicsFunction):

    def __init__(self, model_dir: pathlib.Path, batch_size: int, scenario: ExperimentScenario):
        super().__init__(model_dir, batch_size, scenario)
        self.beta = self.hparams['beta']
        self.batch_size = batch_size
        self.B = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32) * self.beta
        self.states_keys = self.hparams['states_keys']

    def propagate_differentiable(self,
                                 full_env: np.ndarray,
                                 full_env_origin: np.ndarray,
                                 res: float,
                                 start_states: Dict,
                                 actions: tf.Variable) -> List[Dict]:
        """
        :param full_env:        (H, W)
        :param full_env_origin: (2)
        :param res:             scalar
        :param start_states:          each value in the dictionary should be of shape (batch, n_state)
        :param actions:        (T, 2)
        :return: states:       each value in the dictionary should be a of shape [batch, T+1, n_state)
        """

        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        s_t = {}
        for k, s_0_k in start_states.items():
            s_t[k] = tf.convert_to_tensor(s_0_k, dtype=tf.float32)
        predictions = [s_t]
        for t in range(actions.shape[0]):
            action_t = actions[t]

            s_t_plus_1 = {}
            for k, s_t_k in s_t.items():
                n_points = n_state_to_n_points(s_t_k.shape[0])
                delta_s_t = tf.tensordot(action_t, self.B, axes=1)
                delta_s_t_flat = tf.tile(delta_s_t, [n_points])
                s_t_k = s_t_k + delta_s_t_flat * self.dt
                s_t_plus_1[k] = s_t_k

            predictions.append(s_t_plus_1)
            s_t = s_t_plus_1
        return predictions
