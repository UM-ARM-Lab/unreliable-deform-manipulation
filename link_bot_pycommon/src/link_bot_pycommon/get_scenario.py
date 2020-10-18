from link_bot_pycommon.dual_arm_real_victor_rope_scenario import DualArmRealVictorRopeScenario
from link_bot_pycommon.dual_arm_scenario import DualArmScenario
from link_bot_pycommon.dual_arm_sim_rope_scenario import SimDualArmRopeScenario
from link_bot_pycommon.floating_rope_scenario import FloatingRopeScenario
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.rope_dragging_scenario import RopeDraggingScenario

scenario_map = {
    'link_bot': RopeDraggingScenario,
    'rope dragging': RopeDraggingScenario,
    'rope_dragging': RopeDraggingScenario,
    'dragging': RopeDraggingScenario,
    'dual_arm_real_victor': DualArmRealVictorRopeScenario,
    'dual_arm_rope_sim_victor': SimDualArmRopeScenario,
    'dual_arm_rope': SimDualArmRopeScenario,
    'dual_floating_gripper_rope': FloatingRopeScenario,
    'dual_floating': FloatingRopeScenario,
    'dual_arm_no_rope': DualArmScenario,
}


def get_scenario(scenario_name: str) -> ExperimentScenario:
    if scenario_name not in scenario_map:
        raise NotImplementedError(scenario_name)
    return scenario_map[scenario_name]()
