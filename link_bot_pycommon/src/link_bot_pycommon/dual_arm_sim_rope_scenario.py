from typing import Dict

import numpy as np

import rospy
from arm_robots.victor import Victor
from gazebo_ros_link_attacher.srv import AttachRequest
from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from peter_msgs.srv import ExcludeModelsRequest, SetDualGripperPointsRequest


def attach_or_detach_requests():
    robot_name = rospy.get_namespace().strip("/")

    left_req = AttachRequest()
    left_req.model_name_1 = robot_name
    left_req.link_name_1 = "left_tool_placeholder"
    left_req.model_name_2 = "rope_3d"
    left_req.link_name_2 = "left_gripper"
    left_req.anchor_position.x = 0
    left_req.anchor_position.y = 0
    left_req.anchor_position.z = 0
    left_req.has_anchor_position = True

    right_req = AttachRequest()
    right_req.model_name_1 = robot_name
    right_req.link_name_1 = "right_tool_placeholder"
    right_req.model_name_2 = "rope_3d"
    right_req.link_name_2 = "right_gripper"
    right_req.anchor_position.x = 0
    right_req.anchor_position.y = 0
    right_req.anchor_position.z = 0
    right_req.has_anchor_position = True

    return left_req, right_req


class SimDualArmRopeScenario(BaseDualArmRopeScenario):

    def __init__(self):
        super().__init__()
        self.victor = Victor()

    def on_before_data_collection(self, params: Dict):
        # Mark the rope as a not-obstacle
        exclude = ExcludeModelsRequest()
        exclude.model_names.append("rope_3d")
        self.exclude_from_planning_scene_srv(exclude)

        # let go
        self.robot.open_left_gripper()
        self.robot.open_right_gripper()
        self.detach_rope_to_grippers()

        # move to init positions
        self.robot.plan_to_joint_config("both_arms", params['reset_joint_config'])

        # Grasp the rope and move to a certain position to start data collection
        self.service_provider.pause()
        self.move_rope_to_match_grippers()
        self.attach_rope_to_grippers()
        self.service_provider.play()

        self.robot.close_left_gripper()
        self.robot.close_right_gripper()

    def attach_rope_to_grippers(self):
        left_req, right_req = attach_or_detach_requests()
        self.attach_srv(left_req)
        self.attach_srv(right_req)

    def detach_rope_to_grippers(self):
        left_req, right_req = attach_or_detach_requests()
        self.detach_srv(left_req)
        self.detach_srv(right_req)

    def move_rope_to_match_grippers(self):
        left_transform = self.tf.get_transform("robot_root", "left_tool_placeholder")
        right_transform = self.tf.get_transform("robot_root", "right_tool_placeholder")
        desired_rope_point_positions = np.stack([left_transform[0:3, 3], right_transform[0:3, 3]], axis=0)
        move = SetDualGripperPointsRequest()
        move.left_gripper.x = desired_rope_point_positions[0, 0]
        move.left_gripper.y = desired_rope_point_positions[0, 1]
        move.left_gripper.z = desired_rope_point_positions[0, 2]
        move.right_gripper.x = desired_rope_point_positions[1, 0]
        move.right_gripper.y = desired_rope_point_positions[1, 1]
        move.right_gripper.z = desired_rope_point_positions[1, 2]
        self.set_rope_end_points_srv(move)
