import json
import pathlib

import numpy as np

from link_bot_planning.params import LocalEnvParams, FullEnvParams


class BaseForwardModel:

    def __init__(self, model_dir: pathlib.Path):
        model_hparams_file = model_dir / 'hparams.json'
        self.hparams = json.load(model_hparams_file.open('r'))
        self.n_state = self.hparams['dynamics_dataset_hparams']['n_state']
        # TODO: de-duplicate n_action and n_control
        self.n_control = self.hparams['dynamics_dataset_hparams']['n_action']
        self.local_env_params = LocalEnvParams.from_json(self.hparams['dynamics_dataset_hparams']['local_env_params'])
        if 'full_env_params' in self.hparams['dynamics_dataset_hparams']:
            self.full_env_params = FullEnvParams.from_json(self.hparams['dynamics_dataset_hparams']['full_env_params'])
        self.dt = self.hparams['dynamics_dataset_hparams']['dt']
        self.max_step_size = self.hparams['dynamics_dataset_hparams']['max_step_size']

    def predict(self,
                full_envs: np.ndarray,
                full_env_origins: np.ndarray,
                resolution_s: np.ndarray,
                state: np.ndarray,
                actions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
