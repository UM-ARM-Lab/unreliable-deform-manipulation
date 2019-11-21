import json

import numpy as np

from link_bot_planning.params import LocalEnvParams
from link_bot_pycommon import link_bot_sdf_utils


class BaseForwardModel:

    def __init__(self, model_dir):
        model_hparams_file = model_dir / 'hparams.json'
        self.hparams = json.load(model_hparams_file.open('r'))
        self.n_state = self.hparams['dynamics_dataset_hparams']['n_state']
        # TODO: de-duplicate n_action and n_control
        self.n_control = self.hparams['dynamics_dataset_hparams']['n_action']
        self.local_env_params = LocalEnvParams.from_json(self.hparams['dynamics_dataset_hparams']['local_env_params'])
        self.dt = self.hparams['dynamics_dataset_hparams']['dt']

    def predict(self, local_env_data: link_bot_sdf_utils.OccupancyData, first_states: np.ndarray,
                actions: np.ndarray) -> np.ndarray:
        """
        It's T+1 because it includes the first state
        :param local_env_data: local environment
        :param first_states: [batch, 6]
        :param actions: [batch, T, 2]
        :return: [batch, T+1, 3, 2] includes the initial state
        """
        raise NotImplementedError()
