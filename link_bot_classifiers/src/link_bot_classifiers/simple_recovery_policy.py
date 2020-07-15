from typing import Dict
import json
import numpy as np
import pathlib

from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_classifiers.base_recovery_policy import BaseRecoveryPolicy


class SimpleRecoveryPolicy(BaseRecoveryPolicy):

    def __init__(self, hparams: Dict, model_dir: pathlib.Path, scenario: ExperimentScenario):
        super().__init__(hparams, model_dir, scenario)
        self.model_dir = model_dir
        with (self.model_dir / 'action.json').open('r') as action_file:
            self.action = json.load(action_file)
        self.scenario = scenario

    def __call__(self, environment: Dict, state: Dict):
        gripper1_delta = np.array(self.action['gripper1_delta'], dtype=np.float32)
        gripper2_delta = np.array(self.action['gripper2_delta'], dtype=np.float32)

        action = {
            'gripper1_position': state['gripper1'] + gripper1_delta,
            'gripper2_position': state['gripper2'] + gripper2_delta,
        }
        return action
