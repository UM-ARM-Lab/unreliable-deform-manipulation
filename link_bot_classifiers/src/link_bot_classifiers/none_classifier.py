from typing import List

import numpy as np

from link_bot_classifiers.base_classifier import BaseClassifier


class NoneClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()

    def predict(self, local_env_data: List, s1: np.ndarray, s2: np.ndarray, action: np.ndarray) -> List[float]:
        batch_size = len(local_env_data)
        return [1.0] * batch_size
