from typing import Dict

from link_bot_pycommon.dual_floating_gripper_scenario import DualFloatingGripperRopeScenario
from link_bot_pycommon.experiment_scenario import ExperimentScenario


def get_scenario(scenario_name: str) -> ExperimentScenario:
    if scenario_name == 'link_bot':
        raise NotImplementedError()
    elif scenario_name == 'dual_floating_gripper_rope':
        return DualFloatingGripperRopeScenario()
    elif scenario_name == 'dual':
        return DualFloatingGripperRopeScenario()
    else:
        raise NotImplementedError()
