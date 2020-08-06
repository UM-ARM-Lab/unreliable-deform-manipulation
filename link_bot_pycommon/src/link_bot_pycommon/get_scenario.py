from link_bot_pycommon.dual_arm_rope_scenario import DualArmRopeScenario
from link_bot_pycommon.dual_floating_gripper_scenario import DualFloatingGripperRopeScenario
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.rope_dragging_scenario import RopeDraggingScenario


def get_scenario(scenario_name: str) -> ExperimentScenario:
    if scenario_name == 'link_bot':
        return RopeDraggingScenario()
    elif scenario_name == 'rope dragging':
        return RopeDraggingScenario()
    elif scenario_name == 'dragging':
        return RopeDraggingScenario()
    elif scenario_name == 'dual_arm':
        return DualArmRopeScenario()
    elif scenario_name == 'dual_arm_rope':
        return DualArmRopeScenario()
    elif scenario_name == 'dual_floating':
        return DualFloatingGripperRopeScenario()
    else:
        raise NotImplementedError(scenario_name)
