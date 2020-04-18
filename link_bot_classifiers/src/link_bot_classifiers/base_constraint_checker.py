from typing import Dict, List, Callable

import numpy as np
import tensorflow as tf

from link_bot_planning.experiment_scenario import ExperimentScenario


class BaseConstraintChecker:

    def __init__(self, scenario: ExperimentScenario):
        self.scenario = scenario
        self.model_hparams = {}
        self.full_env_params = None

    def check_constraint(self,
                         environement: Dict,
                         states_sequence: List[Dict],
                         actions: np.ndarray) -> float:
        pass

    def check_constraint_differentiable(self,
                                        environment: Dict,
                                        states_sequence: List[Dict],
                                        actions: tf.Variable) -> tf.Tensor:
        pass
