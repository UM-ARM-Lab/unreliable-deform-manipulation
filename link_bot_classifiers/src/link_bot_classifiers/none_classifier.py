from typing import Dict, List

import numpy as np
import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker


class NoneClassifier(BaseConstraintChecker):

    def __init__(self):
        super().__init__()

    def check_constraint(self,
                         full_env: np.ndarray,
                         full_env_origin: np.ndarray,
                         res: float,
                         states_sequence: List[Dict],
                         actions: np.ndarray) -> float:
        return 1.0

    def check_constraint_differentiable(self,
                                        full_env: np.ndarray,
                                        full_env_origin: np.ndarray,
                                        res: float,
                                        states_sequence: List[Dict],
                                        actions: tf.Variable) -> tf.Tensor:
        return tf.ones([], dtype=tf.float32)


model = NoneClassifier
