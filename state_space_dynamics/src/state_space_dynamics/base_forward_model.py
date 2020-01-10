import json
from typing import List

import numpy as np

from link_bot_planning.params import LocalEnvParams


class BaseForwardModel:

    def __init__(self, model_dir):
        model_hparams_file = model_dir / 'hparams.json'
        self.hparams = json.load(model_hparams_file.open('r'))
        self.n_state = self.hparams['dynamics_dataset_hparams']['n_state']
        # TODO: de-duplicate n_action and n_control
        self.n_control = self.hparams['dynamics_dataset_hparams']['n_action']
        self.local_env_params = LocalEnvParams.from_json(self.hparams['dynamics_dataset_hparams']['local_env_params'])
        if 'full_env_params' in self.hparams['dynamics_dataset_hparams']:
            self.full_env_params = LocalEnvParams.from_json(self.hparams['dynamics_dataset_hparams']['full_env_params'])
        self.dt = self.hparams['dynamics_dataset_hparams']['dt']

    def predict(self, local_env_data: List, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        :param local_env_data: [batch] sized list of the local environment data
        :param state: [batch, n_state]
        :param actions: [batch, T, 2]
        :return: [batch, T+1, 3, 2] includes the initial state. It's T+1 because it includes the first state
        """
        raise NotImplementedError()
