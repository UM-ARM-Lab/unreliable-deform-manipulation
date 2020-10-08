from link_bot_pycommon.dual_arm_real_victor_rope_scenario import DualArmRealVictorRopeScenario
from link_bot_pycommon.dual_arm_scenario import DualArmScenario
from link_bot_pycommon.dual_arm_sim_rope_scenario import SimDualArmRopeScenario
from link_bot_pycommon.floating_rope_scenario import FloatingRopeScenario
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
    elif scenario_name == 'dual_arm_real_victor':
        return DualArmRealVictorRopeScenario()
    elif scenario_name == 'dual_arm_rope':
        return SimDualArmRopeScenario()
    elif scenario_name == 'dual_arm_rope':
        return DualArmRealVictorRopeScenario()
    elif scenario_name == 'dual_floating_gripper_rope':
        return FloatingRopeScenario()
    elif scenario_name == 'dual_floating':
        return FloatingRopeScenario()
    elif scenario_name == 'dual_arm_no_rope':
        return DualArmScenario()
    else:
        raise NotImplementedError(scenario_name)
