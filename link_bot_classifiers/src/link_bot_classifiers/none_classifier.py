from typing import Dict, List

import numpy as np
import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario


class NoneClassifier(BaseConstraintChecker):

    def __init__(self, scenario: ExperimentScenario):
        super().__init__(scenario)
        self.horizon = 2

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: np.ndarray):
        return [1.0]

    def check_constraint_differentiable(self,
                                        environment: Dict,
                                        states_sequence: List[Dict],
                                        actions: tf.Variable) -> tf.Tensor:
        return tf.ones([], dtype=tf.float32)


model = NoneClassifier
