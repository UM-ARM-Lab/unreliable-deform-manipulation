from typing import List

import numpy as np

from link_bot_classifiers.base_classifier import BaseClassifier


class NoneClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()

    def predict(self, local_env_data: List, s1_s: np.ndarray, s2_s: np.ndarray) -> List[float]:
        batch_size = len(local_env_data)
        return [1.0] * batch_size

    def predict_state_only(self, local_env_data_s: List, s1_s: np.ndarray) -> List[float]:
        del s1_s  # unused
        batch_size = len(local_env_data_s)
        return [1.0] * batch_size
