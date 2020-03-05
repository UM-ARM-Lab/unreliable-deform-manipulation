from typing import Dict, List, Callable

import numpy as np
import tensorflow as tf


class BaseConstraintChecker:

    def __init__(self, get_local_environment_center: Callable):
        self.get_local_environment_center = get_local_environment_center
        self.model_hparams = {}
        self.full_env_params = None

    def check_constraint(self,
                         full_env: np.ndarray,
                         full_env_origin: np.ndarray,
                         res: float,
                         states_sequence: List[Dict],
                         actions: np.ndarray) -> float:
        pass

    def check_constraint_differentiable(self,
                                        full_env: np.ndarray,
                                        full_env_origin: np.ndarray,
                                        res: float,
                                        states_sequence: List[Dict],
                                        actions: tf.Variable) -> tf.Tensor:
        pass
