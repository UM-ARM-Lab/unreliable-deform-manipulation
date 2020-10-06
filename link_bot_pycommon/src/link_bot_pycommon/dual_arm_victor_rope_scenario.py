from typing import Dict

from arm_robots.victor import Victor
from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from victor_hardware_interface_msgs.msg import ControlMode


class DualArmVictorRopeScenario(BaseDualArmRopeScenario):

    def __init__(self):
        super().__init__()
        self.victor = Victor()

    def on_before_data_collection(self, params: Dict):
        self.victor.base_victor.set_control_mode(ControlMode.JOINT_IMPEDANCE)
        super().on_before_data_collection(params)
