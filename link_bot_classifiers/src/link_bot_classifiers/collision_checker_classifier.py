import json
import pathlib
from typing import List, Dict

import numpy as np
import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.collision_checking import batch_in_collision_tf
from link_bot_pycommon.experiment_scenario import ExperimentScenario

DEFAULT_INFLATION_RADIUS = 0.02


class CollisionCheckerClassifier(BaseConstraintChecker):

    def __init__(self, path: pathlib.Path, inflation_radius: float, scenario: ExperimentScenario):
        super().__init__(scenario)
        self.inflation_radius = inflation_radius
        hparams_file = path / 'hparams.json'
        self.model_hparams = json.load(hparams_file.open('r'))
        self.local_h_rows = self.model_hparams['local_h_rows']
        self.local_w_cols = self.model_hparams['local_w_cols']
        self.horizon = 2

    def check_constraint_differentiable(self,
                                        environment: Dict,
                                        states_sequence: List[Dict],
                                        actions) -> tf.Tensor:
        state = states_sequence[-1]
        if self.model_hparams['collision_check_object']:
            points = self.scenario.state_to_points(state)
        else:
            points = self.scenario.state_to_gripper_position(state)
        xs = points[:, 0]
        ys = points[:, 1]
        in_collision = batch_in_collision_tf(environment=environment,
                                             xs=xs,
                                             ys=ys)
        prediction = 1 - in_collision
        return prediction

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: np.ndarray):
        return self.check_constraint_differentiable(environment, states_sequence, actions).numpy()


model = CollisionCheckerClassifier
