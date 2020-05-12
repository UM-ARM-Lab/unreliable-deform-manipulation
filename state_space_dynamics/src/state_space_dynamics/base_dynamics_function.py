import json
import pathlib
from typing import Dict, List

import numpy as np
import tensorflow as tf

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_planning.params import FullEnvParams, SimParams


class BaseDynamicsFunction:

    def __init__(self, model_dir: pathlib.Path, batch_size: int, scenario: ExperimentScenario):
        model_hparams_file = model_dir / 'hparams.json'
        self.scenario = scenario
        self.hparams = json.load(model_hparams_file.open('r'))
        self.batch_size = batch_size
        self.n_action = self.hparams['dynamics_dataset_hparams']['n_action']
        self.sim_params = SimParams.from_json(self.hparams['dynamics_dataset_hparams']['sim_params'])
        if 'full_env_params' in self.hparams['dynamics_dataset_hparams']:
            self.full_env_params = FullEnvParams.from_json(self.hparams['dynamics_dataset_hparams']['full_env_params'])
        self.dt = self.hparams['dynamics_dataset_hparams']['dt']
        self.max_step_size = self.sim_params.max_step_size
        self.states_description = self.hparams['dynamics_dataset_hparams']['states_description']
        self.states_keys = None

    def propagate_from_dataset_element(self, dataset_element):
        raise NotImplementedError()

    def propagate(self,
                  full_env: np.ndarray,
                  full_env_origin: np.ndarray,
                  res: float,
                  start_states: Dict[str, np.ndarray],
                  actions: np.ndarray) -> List[Dict]:
        for k in start_states.keys():
            start_states[k] = start_states[k].astype(np.float32)
        actions = actions.astype(np.float32)

        actions = tf.Variable(actions, dtype=tf.float32, name='actions')
        predictions = self.propagate_differentiable(full_env,
                                                    full_env_origin,
                                                    res,
                                                    start_states,
                                                    actions)
        predictions_np = []
        for state_t in predictions:
            state_np = {}
            for k, v in state_t.items():
                state_np[k] = v.numpy().astype(np.float64)
            predictions_np.append(state_np)

        return predictions_np

    def propagate_differentiable(self,
                                 full_env: np.ndarray,
                                 full_env_origin: np.ndarray,
                                 res: float,
                                 start_states: Dict[str, np.ndarray],
                                 actions: tf.Variable) -> List[Dict]:
        raise NotImplementedError()
