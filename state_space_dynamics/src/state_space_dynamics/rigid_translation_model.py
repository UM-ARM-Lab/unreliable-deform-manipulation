import pathlib
from typing import Optional

import numpy as np

from state_space_dynamics.base_forward_model import BaseForwardModel
from link_bot_pycommon import link_bot_sdf_utils


class RigidTranslationModel(BaseForwardModel):

    def __init__(self, model_dir: pathlib.Path, beta: Optional[float] = None, dt: Optional[float] = None):
        super().__init__(model_dir)
        if beta is None:
            self.beta = self.hparams['beta']
        else:
            self.beta = beta

        if dt is None:
            self.dt = self.hparams['dt']
        else:
            self.dt = dt

    def predict(self, local_env_data: link_bot_sdf_utils.OccupancyData, state: np.ndarray,
                actions: np.ndarray) -> np.ndarray:
        del local_env_data  # unused
        predictions = []
        for state, actions in zip(state, actions):
            s_0 = np.reshape(state, [3, 2])
            prediction = [s_0]
            s_t = s_0
            for action in actions:
                # I've tuned beta on the no_obj_new training set based on the total error
                B = np.tile(np.eye(2), [3, 1])
                s_t = s_t + np.reshape(B @ action, [3, 2]) * self.dt * self.beta
                prediction.append(s_t)
            prediction = np.array(prediction)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions
