from typing import List

import numpy as np

from link_bot_classifiers.base_classifier import BaseClassifier


class NoneClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()

    def predict(self, local_env_datas: List, s1_s: np.ndarray, s2_s: np.ndarray) -> float:
        return 1.0

    def predict_state_only(self, local_env_datas: List, s1_s: np.ndarray) -> float:
        return 1.0
