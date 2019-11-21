import numpy as np

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_pycommon import link_bot_sdf_utils


class NoneClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()

    def predict(self, local_env_data: link_bot_sdf_utils.OccupancyData, s1: np.ndarray, s2: np.ndarray) -> float:
        return 1.0

    def predict_state_only(self, local_env_data: link_bot_sdf_utils.OccupancyData, s1: np.ndarray) -> float:
        return 1.0
