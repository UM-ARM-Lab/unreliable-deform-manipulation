from typing import List

import numpy as np

from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.link_bot_sdf_utils import OccupancyData


class BaseClassifier:

    def __init__(self):
        self.model_hparams = {}

    def predict(self, local_env_data: OccupancyData, s1: np.ndarray, s2: np.ndarray, action: np.ndarray) -> float:
        pass

    def predict_traj(self, full_env: OccupancyData, states: np.ndarray, actions: np.ndarray) -> float:
        pass
