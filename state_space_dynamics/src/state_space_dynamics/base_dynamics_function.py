import json
import pathlib
from typing import Dict

import numpy as np

from link_bot_planning.params import LocalEnvParams, FullEnvParams, SimParams


class BaseDynamicsFunction:

    def __init__(self, model_dir: pathlib.Path, batch_size: int):
        model_hparams_file = model_dir / 'hparams.json'
        self.hparams = json.load(model_hparams_file.open('r'))
        self.batch_size = batch_size
        self.n_action = self.hparams['dynamics_dataset_hparams']['n_action']
        self.local_env_params = LocalEnvParams.from_json(self.hparams['dynamics_dataset_hparams']['local_env_params'])
        self.sim_params = SimParams.from_json(self.hparams['dynamics_dataset_hparams']['sim_params'])
        if 'full_env_params' in self.hparams['dynamics_dataset_hparams']:
            self.full_env_params = FullEnvParams.from_json(self.hparams['dynamics_dataset_hparams']['full_env_params'])
        self.dt = self.hparams['dynamics_dataset_hparams']['dt']
        self.max_step_size = self.sim_params.max_step_size
        self.states_description = self.hparams['dynamics_dataset_hparams']['states_description']
        self.state_keys = None

    def propagate(self,
                  full_env: np.ndarray,
                  full_env_origin: np.ndarray,
                  res: np.ndarray,
                  states: Dict[str, np.ndarray],
                  actions: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    def propagate_differentiable(self,
                                 full_env: np.ndarray,
                                 full_env_origin: np.ndarray,
                                 res: np.ndarray,
                                 states: Dict[str, np.ndarray],
                                 actions: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError()
