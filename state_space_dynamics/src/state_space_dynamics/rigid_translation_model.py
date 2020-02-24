import pathlib

import numpy as np

from state_space_dynamics.base_forward_model import BaseForwardModel


class RigidTranslationModel(BaseForwardModel):

    def __init__(self, model_dir: pathlib.Path):
        super().__init__(model_dir)
        self.beta = self.hparams['beta']

    def predict(self,
                full_env: np.ndarray,
                full_env_origin: np.ndarray,
                res: np.ndarray,
                state: np.ndarray,
                actions: np.ndarray) -> np.ndarray:
        predictions = []
        for state, actions in zip(state, actions):
            s_0 = np.reshape(state, [-1, 2])
            prediction = [s_0]
            s_t = s_0
            for action in actions:
                # I've tuned beta on the no_obj_new training set based on the total error
                B = np.tile(np.eye(2), [self.n_points, 1])
                s_t = s_t + np.reshape(B @ action, [-1, 2]) * self.dt * self.beta
                prediction.append(s_t)
            prediction = np.array(prediction)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions
