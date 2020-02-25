from typing import Dict

import numpy as np

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_pycommon.link_bot_sdf_utils import OccupancyData


class NoneClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()

    def predict_transition(self, local_env_data: OccupancyData, s1: np.ndarray, s2: np.ndarray, action: np.ndarray) -> float:
        return 1.0

    def predict_traj(self, full_env: OccupancyData, states: Dict[str, np.ndarray], actions: np.ndarray) -> float:
        return 1.0

    def predict(self, full_env: OccupancyData, states: Dict[str, np.ndarray], actions: np.ndarray) -> float:
        return 1.0
