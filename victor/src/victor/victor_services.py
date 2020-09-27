import rospy
from arc_utilities.ros_helpers import TF2Wrapper
from arm_robots.victor import Victor
from link_bot_pycommon.base_services import BaseServices
from peter_msgs.srv import GetDualGripperPoints, GetDualGripperPointsRequest, GetDualGripperPointsResponse
from victor_hardware_interface_msgs.msg import ControlMode

gripper_closed_positions = [
    1.1,
    -0.3,
    -0.60,
    1.1,
    -0.3,
    -0.60,
    0.175,  # 0.175 is the max inwardness for the knuckle
    2.0,
    1.5,
    2.0,
    -0.175,
]

class VictorServices(BaseServices):
    def __init__(self):
        super().__init__()
        self.victor = Victor()
        self.gripper_points_service = rospy.Service("get_dual_gripper_points",
                                                    GetDualGripperPoints,
                                                    self.get_dual_gripper_points_cb)
        self.tf_wrapper = TF2Wrapper()

    def get_dual_gripper_points_cb(self, req: GetDualGripperPointsRequest):
        del req  # unused
        res = GetDualGripperPointsResponse()
        left_tool_transform = self.tf_wrapper.get_transform(parent="victor_root",
                                                            child="left_tool_placeholder")
        res.gripper1.x = left_tool_transform[0, 3]
        res.gripper1.y = left_tool_transform[1, 3]
        res.gripper1.z = left_tool_transform[2, 3]
        right_tool_transform = self.tf_wrapper.get_transform(parent="victor_root",
                                                             child="right_tool_placeholder")
        res.gripper2.x = right_tool_transform[0, 3]
        res.gripper2.y = right_tool_transform[1, 3]
        res.gripper2.z = right_tool_transform[2, 3]
        return res

    def setup_env(self, verbose: int, real_time_rate: float, max_step_size: float):
        # set the robot into impedance mode
        success = self.victor.set_control_mode(ControlMode.JOINT_IMPEDANCE)
        if not success:
            raise RuntimeError("Failed to switch into impedance mode")
        # self.victor.set_right_gripper(gripper_closed_positions)
        # self.victor.set_left_gripper(gripper_closed_positions)
        pass
