import pathlib
from typing import Dict, List

import numpy as np
import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario


class NoneClassifier(BaseConstraintChecker):

    def __init__(self, paths: List[pathlib.Path], scenario: ExperimentScenario):
        super().__init__(paths, scenario)
        self.horizon = 2
        self.data_collection_params = {
            'res': 0.02,
        }

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: np.ndarray):
        return [1.0], [1e-9]

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions: tf.Variable):
        return tf.ones([], dtype=tf.float32), tf.ones([], dtype=tf.float32) * 1e-9


model = NoneClassifier
