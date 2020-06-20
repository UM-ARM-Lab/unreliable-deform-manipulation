from typing import Dict

from link_bot_pycommon.dual_floating_gripper_scenario import DualFloatingGripperRopeScenario
from link_bot_pycommon.fishing_3d_scenario import Fishing3DScenario
from link_bot_pycommon.link_bot_scenario import LinkBotScenario


def get_scenario(scenario_name: str, params: Dict):
    if scenario_name == 'link_bot':
        return LinkBotScenario(params)
    elif scenario_name == 'dual_floating_gripper_rope':
        return DualFloatingGripperRopeScenario(params)
    elif scenario_name == 'fishing':
        return Fishing3DScenario(params)
    else:
        raise NotImplementedError()
