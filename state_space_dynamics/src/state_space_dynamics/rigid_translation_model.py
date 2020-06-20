import pathlib
from typing import Dict, List

import numpy as np
import tensorflow as tf

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.pycommon import n_state_to_n_points
from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts, sequence_of_dicts_to_dict_of_sequences
from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class RigidTranslationModel(BaseDynamicsFunction):

    def __init__(self, model_dir: pathlib.Path, batch_size: int, scenario: ExperimentScenario):
        super().__init__(model_dir, batch_size, scenario)
        self.batch_size = batch_size
        b = self.hparams['B']
        self.B = tf.constant(np.array(b), dtype=tf.float32)
        self.states_keys = self.hparams['states_keys']

    def propagate_from_example(self, dataset_element):
        inputs, _ = dataset_element
        batch_states = {key: inputs[key] for key in self.states_keys}
        batch_states = dict_of_sequences_to_sequence_of_dicts(batch_states)
        predictions = {k: [] for k in self.states_keys}
        for full_env, full_env_origin, res, states, actions in zip(inputs['full_env/env'],
                                                                   inputs['full_env/origin'],
                                                                   inputs['full_env/res'],
                                                                   batch_states,
                                                                   inputs['action']):
            start_states = {k: states[k][0] for k in states.keys()}
            out_states = self.propagate_differentiable(full_env=full_env,
                                                       full_env_origin=full_env_origin,
                                                       res=res,
                                                       start_states=start_states,
                                                       actions=actions)
            out_states = sequence_of_dicts_to_dict_of_sequences(out_states)
            for k in self.states_keys:
                predictions[k].append(out_states[k])
        predictions = {k: tf.stack(predictions[k], axis=0) for k in predictions.keys()}
        return predictions

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
                delta_s_t = tf.tensordot(action_t, tf.transpose(self.B), axes=1)
                delta_s_t_flat = tf.tile(delta_s_t, [n_points])
                s_t_k = s_t_k + delta_s_t_flat * self.dt
                s_t_plus_1[k] = s_t_k

            predictions.append(s_t_plus_1)
            s_t = s_t_plus_1
        return predictions
