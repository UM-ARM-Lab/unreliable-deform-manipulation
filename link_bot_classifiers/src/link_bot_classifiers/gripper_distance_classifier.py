import json
import pathlib
from typing import List, Dict

import numpy as np
import tensorflow as tf

from link_bot_classifiers.base_constraint_checker import BaseConstraintChecker
from link_bot_pycommon.experiment_scenario import ExperimentScenario


class GripperDistanceClassifier(BaseConstraintChecker):

    def __init__(self,
                 paths: List[pathlib.Path],
                 scenario: ExperimentScenario,
                 ):
        super().__init__(paths, scenario)
        assert len(paths) == 1
        self.path = paths[0]
        hparams_file = self.path.parent / 'params.json'
        self.hparams = json.load(hparams_file.open('r'))
        self.horizon = 2
        self.max_d = self.hparams['max_distance_between_grippers']

    def check_constraint_tf(self,
                            environment: Dict,
                            states_sequence: List[Dict],
                            actions):
        del environment  # unused
        assert len(states_sequence) == 2
        not_too_far = tf.linalg.norm(states_sequence[1]['gripper2'] - states_sequence[1]['gripper1']) < self.max_d
        return tf.expand_dims(tf.cast(not_too_far, tf.float32), axis=0), None

    def check_constraint(self,
                         environment: Dict,
                         states_sequence: List[Dict],
                         actions: List[Dict]):
        del environment  # unused
        assert len(states_sequence) == 2
        d = np.linalg.norm(states_sequence[1]['gripper2'] - states_sequence[1]['gripper1'])
        not_too_far = d < self.max_d
        return [not_too_far.astype(np.float32)], None
