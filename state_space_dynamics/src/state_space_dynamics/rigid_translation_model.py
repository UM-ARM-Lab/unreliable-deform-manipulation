import pathlib
from typing import Dict

import numpy as np

from state_space_dynamics.base_dynamics_function import BaseDynamicsFunction


class RigidTranslationModel(BaseDynamicsFunction):

    def __init__(self, model_dir: pathlib.Path):
        super().__init__(model_dir)
        self.beta = self.hparams['beta']

    def propagate(self,
                  full_env: np.ndarray,
                  full_env_origin: np.ndarray,
                  res: np.ndarray,
                  states: Dict[str, np.ndarray],
                  actions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        :param full_env:        (H, W)
        :param full_env_origin: (2)
        :param res:     scalar
        :param states:          each value in the dictionary should be of shape (batch, n_state)
        :param actions:         (T, 2)
        :return: states:         each value in the dictionary should be a of shape [batch, T+1, n_state)
        """
        state = states['link_bot']
        prediction = [state]
        s_0 = np.reshape(state, [-1, 2])
        s_t = s_0
        for action in actions:
            B = np.tile(np.eye(2), [self.n_points, 1])
            s_t = s_t + np.reshape(B @ action, [-1, 2]) * self.dt * self.beta
            prediction.append(s_t.squeeze())
        prediction = np.array(prediction)
        return {'link_bot': prediction}
