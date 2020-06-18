from typing import Dict, Optional

import numpy as np

import rospy
from geometry_msgs.msg import Pose
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from link_bot_pycommon.params import CollectDynamicsParams
from link_bot_pycommon.pycommon import quaternion_from_euler
from mps_shape_completion_msgs.msg import OccupancyStamped
from peter_msgs.srv import Position3DAction, Position3DActionRequest, SetRopeConfiguration, Position3DEnableRequest, \
    SetRopeConfigurationRequest
from std_srvs.srv import Empty
from visualization_msgs.msg import MarkerArray


class Fishing3DScenario(Base3DScenario):
    def __init__(self):
        super().__init__()
        object_name = 'link_bot'
        self.set_srv = rospy.ServiceProxy(f"{object_name}/set", Position3DAction)
        self.stop_object_srv = rospy.ServiceProxy(f"{object_name}/stop", Empty)
        self.object_enable_srv = rospy.ServiceProxy(f"{object_name}/enable", Empty)
        self.get_object_srv = rospy.ServiceProxy(f"{object_name}/get", Empty)
        self.set_rope_config_srv = rospy.ServiceProxy("set_rope_config", SetRopeConfiguration)

        self.env_viz_srv = rospy.Publisher('occupancy', OccupancyStamped, queue_size=10)
        self.state_viz_srv = rospy.Publisher("state_viz", MarkerArray, queue_size=10)
        self.action_viz_srv = rospy.Publisher("action_viz", MarkerArray, queue_size=10)

    def enable_object(self):
        enable_object = Position3DEnableRequest()
        enable_object.enable = True
        self.object_enable_srv(enable_object)

    def execute_action(self, action: Dict):
        req = Position3DActionRequest()
        req.position.x = action['position'][0]
        req.position.y = action['position'][1]
        req.position.z = action['position'][2]
        req.timeout = action['timeout'][0]
        # I think Dale's going to write something which takes in a desired point in 3D
        # and computes a joint-space trajectory to send to gazebo. I could write a similar plugin
        # which simply takes a desired point in 3D and linearly interpolates a kinematic link's position
        # along that path

        _ = self.set_srv(req)

    def reset_rope(self, x, y, yaw, joint_angles):
        gripper_pose = Pose()
        gripper_pose.position.x = x
        gripper_pose.position.y = y
        q = quaternion_from_euler(0, 0, yaw)
        gripper_pose.orientation.x = q[0]
        gripper_pose.orientation.y = q[1]
        gripper_pose.orientation.z = q[2]
        gripper_pose.orientation.w = q[3]
        req = SetRopeConfigurationRequest()
        req.gripper_poses.append(gripper_pose)
        req.joint_angles.extend(joint_angles)
        self.set_rope_config_srv(req)

    @staticmethod
    def sample_action(environment: Dict,
                      service_provider,
                      state,
                      last_action: Optional[Dict],
                      params: CollectDynamicsParams,
                      action_rng):
        # sample the previous action with 80% probability, this improves exploration
        if last_action is not None and action_rng.uniform(0, 1) < 0.80:
            return last_action
        else:
            return Fishing3DScenario.random_delta_position_action(state, action_rng, environment)

    def __repr__(self):
        return "Fishing3d"
