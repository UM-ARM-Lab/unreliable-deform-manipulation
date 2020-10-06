from link_bot_pycommon.dual_arm_victor_rope_scenario import DualArmVictorRopeScenario
from link_bot_pycommon.dual_arm_scenario import DualArmScenario
from link_bot_pycommon.dual_floating_gripper_scenario import FloatingRopeScenario
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.rope_dragging_scenario import RopeDraggingScenario


def get_scenario(scenario_name: str) -> ExperimentScenario:
    if scenario_name == 'link_bot':
        return RopeDraggingScenario()
    elif scenario_name == 'rope dragging':
        return RopeDraggingScenario()
    elif scenario_name == 'rope_dragging':
        return RopeDraggingScenario()
    elif scenario_name == 'dragging':
        return RopeDraggingScenario()
    elif scenario_name == 'dual_arm':
        return DualArmVictorRopeScenario()
    elif scenario_name == 'dual_arm_rope':
        return DualArmVictorRopeScenario()
    elif scenario_name == 'dual_floating_gripper_rope':
        return FloatingRopeScenario()
    elif scenario_name == 'dual_floating':
        return FloatingRopeScenario()
    elif scenario_name == 'dual_arm_no_rope':
        return DualArmScenario()
    else:
        raise NotImplementedError(scenario_name)
