from typing import Dict

import numpy as np

import ros_numpy
import rospy
from actionlib import SimpleActionClient, GoalStatus
from control_msgs.msg import FollowJointTrajectoryFeedback, FollowJointTrajectoryGoal
from gazebo_ros_link_attacher.srv import AttachRequest
from link_bot_gazebo_python.gazebo_services import GazeboServices
from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from peter_msgs.srv import ExcludeModelsRequest, SetDualGripperPointsRequest, GetBoolRequest, GetBool, GetBoolResponse, \
    Position3DActionRequest, Position3DAction
from rosgraph.names import ns_join


class SimDualArmRopeScenario(BaseDualArmRopeScenario):

    def __init__(self):
        super().__init__('victor')

        self.service_provider = GazeboServices()

        # register a new callback to stop when the rope is overstretched
        self.overstretching_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "rope_overstretched"), GetBool)
        self.set_rope_end_points_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "set"), Position3DAction)

    def overstretching_stop_condition(self, feedback: FollowJointTrajectoryFeedback):
        res: GetBoolResponse = self.overstretching_srv(GetBoolRequest())
        return res.data

    def execute_action(self, action: Dict):
        start_left_gripper_position, start_right_gripper_position = self.robot.get_gripper_positions()
        left_gripper_points = [action['left_gripper_position']]
        right_gripper_points = [action['right_gripper_position']]
        tool_names = [self.robot.left_tool_name, self.robot.right_tool_name]
        grippers = [left_gripper_points, right_gripper_points]

        def _stop_condition(feedback):
            return self.overstretching_stop_condition(feedback)

        traj, result, state = self.robot.follow_jacobian_to_position(group_name=r"both_arms",
                                                                     tool_names=tool_names,
                                                                     preferred_tool_orientations=None,
                                                                     points=grippers,
                                                                     stop_condition=_stop_condition)

        if state == GoalStatus.PREEMPTED:
            def _rev_stop_condition(feedback):
                return not self.overstretching_stop_condition(feedback)

            rev_grippers = [[ros_numpy.numpify(start_left_gripper_position)],
                            [ros_numpy.numpify(start_right_gripper_position)]]
            self.robot.follow_jacobian_to_position("both_arms",
                                                   tool_names,
                                                   preferred_tool_orientations=None,
                                                   points=rev_grippers,
                                                   stop_condition=_rev_stop_condition)

    def stop_on_rope_overstretching(self,
                                    client: SimpleActionClient,
                                    goal: FollowJointTrajectoryGoal,
                                    feedback: FollowJointTrajectoryFeedback):
        stop = self.overstretching_stop_condition(feedback)
        if stop:
            client.cancel_all_goals()

    def on_before_data_collection(self, params: Dict):
        super().on_before_data_collection(params)

        # Mark the rope as a not-obstacle
        exclude = ExcludeModelsRequest()
        exclude.model_names.append("rope_3d")
        self.exclude_from_planning_scene_srv(exclude)

        # let go
        # TODO: if not grasp:
        #  see the real_victor scenario on_before_data_collection

        # else don't bother
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

        rospy.sleep(2)

        self.robot.close_left_gripper()
        self.robot.close_right_gripper()

    def attach_or_detach_requests(self):
        left_req = AttachRequest()
        # Note: gazebo model name could potentially be different from robot namespace, so this could be wrong...
        left_req.model_name_1 = self.robot_namespace
        left_req.link_name_1 = self.robot.left_tool_name
        left_req.model_name_2 = "rope_3d"
        left_req.link_name_2 = "left_gripper"

        right_req = AttachRequest()
        right_req.model_name_1 = self.robot_namespace
        right_req.link_name_1 = self.robot.right_tool_name
        right_req.model_name_2 = "rope_3d"
        right_req.link_name_2 = "right_gripper"

        return left_req, right_req

    def attach_rope_to_grippers(self):
        set_req = Position3DActionRequest()
        self.set_rope_end_points_srv(set_req)

    def detach_rope_to_grippers(self):
        left_req, right_req = self.attach_or_detach_requests()
        self.detach_srv(left_req)
        self.detach_srv(right_req)

    def move_rope_to_match_grippers(self):
        left_transform = self.tf.get_transform("robot_root", self.robot.left_tool_name)
        right_transform = self.tf.get_transform("robot_root", self.robot.right_tool_name)
        desired_rope_point_positions = np.stack([left_transform[0:3, 3], right_transform[0:3, 3]], axis=0)
        move = SetDualGripperPointsRequest()
        move.left_gripper.x = desired_rope_point_positions[0, 0]
        move.left_gripper.y = desired_rope_point_positions[0, 1]
        move.left_gripper.z = desired_rope_point_positions[0, 2]
        move.right_gripper.x = desired_rope_point_positions[1, 0]
        move.right_gripper.y = desired_rope_point_positions[1, 1]
        move.right_gripper.z = desired_rope_point_positions[1, 2]
        self.set_rope_end_points_srv(move)

    def randomize_environment(self, env_rng: np.random.RandomState, params: Dict):
        # release the rope
        # self.robot.open_left_gripper()
        # self.robot.open_right_gripper()
        # self.detach_rope_to_grippers()

        # plan to reset joint config, we assume this will always work
        # self.robot.plan_to_joint_config("both_arms", params['reset_joint_config'])

        # possibly randomize the obstacle configurations?
        random_object_poses = self.random_new_object_poses(env_rng, params)
        self.set_object_poses(random_object_poses)

        # Grasp the rope again
        # self.service_provider.pause()
        # self.move_rope_to_match_grippers()
        # self.attach_rope_to_grippers()
        # self.service_provider.play()
