from typing import Dict

import numpy as np

import rospy
from link_bot_gazebo_python.gazebo_services import GazeboServices
from link_bot_pycommon.base_dual_arm_rope_scenario import BaseDualArmRopeScenario
from link_bot_pycommon.dual_arm_rope_action import dual_arm_rope_execute_action
from peter_msgs.srv import *
from rosgraph.names import ns_join


def gz_scope(*args):
    return "::".join(args)


class SimDualArmRopeScenario(BaseDualArmRopeScenario):

    def __init__(self):
        super().__init__('victor')

        self.service_provider = GazeboServices()

        # register a new callback to stop when the rope is overstretched
        self.set_rope_end_points_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "set"), Position3DAction)
        self.register_controller_srv = rospy.ServiceProxy("/position_3d_plugin/register", RegisterPosition3DController)
        self.pos3d_follow_srv = rospy.ServiceProxy("/position_3d_plugin/follow", Position3DFollow)
        self.pos3d_enable_srv = rospy.ServiceProxy("/position_3d_plugin/enable", Position3DEnable)
        self.pos3d_set_srv = rospy.ServiceProxy("/position_3d_plugin/set", Position3DAction)

    def execute_action(self, action: Dict):
        return dual_arm_rope_execute_action(self.robot, action)

    def on_before_data_collection(self, params: Dict):
        super().on_before_data_collection(params)

        # register kinematic controllers for fake-grasping
        self.register_fake_grasping()

        # Mark the rope as a not-obstacle
        exclude = ExcludeModelsRequest()
        exclude.model_names.append("rope_3d")
        self.exclude_from_planning_scene_srv(exclude)

        # move to init positions
        self.robot.plan_to_joint_config("both_arms", params['reset_joint_config'])

        # Grasp the rope again
        self.grasp_rope_endpoints()

    def register_fake_grasping(self):
        register_left_req = RegisterPosition3DControllerRequest()
        register_left_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, "left_gripper")
        register_left_req.controller_type = "kinematic"
        self.register_controller_srv(register_left_req)
        register_right_req = RegisterPosition3DControllerRequest()
        register_right_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, "right_gripper")
        register_right_req.controller_type = "kinematic"
        self.register_controller_srv(register_right_req)

    def randomize_environment(self, env_rng: np.random.RandomState, params: Dict):
        # release the rope
        self.release_rope_endpoints()

        # teleport movable objects out of the way
        self.move_objects_out_of_scene(params)

        # plan to reset joint config, we assume this will always work
        self.robot.plan_to_joint_config("both_arms", params['reset_joint_config'])

        # Grasp the rope again
        self.grasp_rope_endpoints()

        # randomize the object configurations
        random_object_poses = self.random_new_object_poses(env_rng, params)
        self.set_object_poses(random_object_poses)

    def grasp_rope_endpoints(self):
        self.robot.open_left_gripper()
        self.robot.open_right_gripper()

        self.service_provider.pause()
        self.make_rope_endpoints_follow_gripper()
        self.service_provider.play()
        rospy.sleep(5)
        self.robot.close_left_gripper()
        self.robot.close_right_gripper()

        self.reset_cdcpd()

    def release_rope_endpoints(self):
        self.robot.open_left_gripper()
        self.robot.open_right_gripper()
        self.detach_rope_from_grippers()

    def move_rope_out_of_the_scene(self):
        set_req = Position3DActionRequest()
        set_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, "left_gripper")
        set_req.position.x = 1.3
        set_req.position.y = 0.3
        set_req.position.z = 1.3
        self.pos3d_set_srv(set_req)

        set_req = Position3DActionRequest()
        set_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, "right_gripper")
        set_req.position.x = 1.3
        set_req.position.y = -0.3
        set_req.position.z = 1.3
        self.pos3d_set_srv(set_req)

    def make_rope_endpoints_follow_gripper(self):
        left_follow_req = Position3DFollowRequest()
        left_follow_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, "left_gripper")
        left_follow_req.frame_id = "left_tool"
        self.pos3d_follow_srv(left_follow_req)

        right_follow_req = Position3DFollowRequest()
        right_follow_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, "right_gripper")
        right_follow_req.frame_id = "right_tool"
        self.pos3d_follow_srv(right_follow_req)

    def detach_rope_from_grippers(self):
        left_enable_req = Position3DEnableRequest()
        left_enable_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, "left_gripper")
        left_enable_req.enable = False
        self.pos3d_enable_srv(left_enable_req)

        right_enable_req = Position3DEnableRequest()
        right_enable_req.scoped_link_name = gz_scope(self.ROPE_NAMESPACE, "right_gripper")
        right_enable_req.enable = False
        self.pos3d_enable_srv(right_enable_req)

    def move_objects_out_of_scene(self, params: Dict):
        position = [0, 2, 0]
        orientation = [0, 0, 0, 1]
        out_of_scene_object_poses = {k: (position, orientation) for k in params['objects']}
        self.set_object_poses(out_of_scene_object_poses)
