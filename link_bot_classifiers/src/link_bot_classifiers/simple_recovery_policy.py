import numpy as np
from typing import Dict
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_classifiers.base_recovery_policy import BaseRecoveryPolicy


class RandomRecoveryPolicy(BaseRecoveryPolicy):

    def __init__(self, scenario: ExperimentScenario):
        self.scenario = scenario

    def __call__(self, environment: Dict, state: Dict):
        gripper1_delta = np.array([-0.01711, 0.05217, 0.07524], dtype=np.float32)
        gripper2_delta = np.array([0.00152, -0.01089, 0.0239], dtype=np.float32)

        action = {
            'gripper1_position': state['gripper1'] + gripper1_delta,
            'gripper2_position': state['gripper2'] + gripper2_delta,
        }
        return action
