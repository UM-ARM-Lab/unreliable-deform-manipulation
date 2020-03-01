import json
import pathlib
from typing import Dict

import numpy as np

from link_bot_planning.params import LocalEnvParams, FullEnvParams, SimParams
from link_bot_pycommon.link_bot_pycommon import n_state_to_n_points


class BaseDynamicsFunction:

    def __init__(self, model_dir: pathlib.Path):
        model_hparams_file = model_dir / 'hparams.json'
        self.hparams = json.load(model_hparams_file.open('r'))
        self.n_state = self.hparams['dynamics_dataset_hparams']['n_state']
        self.n_action = self.hparams['dynamics_dataset_hparams']['n_action']
        self.local_env_params = LocalEnvParams.from_json(self.hparams['dynamics_dataset_hparams']['local_env_params'])
        self.sim_params = SimParams.from_json(self.hparams['dynamics_dataset_hparams']['sim_params'])
        if 'full_env_params' in self.hparams['dynamics_dataset_hparams']:
            self.full_env_params = FullEnvParams.from_json(self.hparams['dynamics_dataset_hparams']['full_env_params'])
        self.dt = self.hparams['dynamics_dataset_hparams']['dt']
        self.max_step_size = self.sim_params.max_step_size
        self.n_points = n_state_to_n_points(self.n_state)

    def propagate(self,
                  full_env: np.ndarray,
                  full_env_origin: np.ndarray,
                  res: np.ndarray,
                  states: Dict[str, np.ndarray],
                  actions: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError()
