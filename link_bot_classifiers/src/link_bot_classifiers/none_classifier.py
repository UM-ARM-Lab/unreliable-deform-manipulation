from typing import List

import numpy as np

from link_bot_classifiers.base_classifier import BaseClassifier
from link_bot_pycommon.link_bot_sdf_utils import OccupancyData


class NoneClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()

    def predict(self, local_env_data: OccupancyData, s1: np.ndarray, s2: np.ndarray, action: np.ndarray) -> List[float]:
        return 1.0
