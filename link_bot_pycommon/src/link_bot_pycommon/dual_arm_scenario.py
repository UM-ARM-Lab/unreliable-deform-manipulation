from typing import Dict

import ros_numpy

import rospy
from arc_utilities.ros_helpers import Listener
from arm_robots_msgs.msg import Points
from arm_robots_msgs.srv import GrippersTrajectoryRequest
from geometry_msgs.msg import Point
from link_bot_gazebo_python.gazebo_services import GazeboServices
from link_bot_pycommon.dual_floating_gripper_scenario import DualFloatingGripperRopeScenario
from peter_msgs.srv import GetDualGripperPointsRequest
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty


class DualArmScenario(DualFloatingGripperRopeScenario):

    def __init__(self):
        super().__init__()
        # TODO: robot_name?
        robot_name = 'victor'
        self.joint_states_listener = Listener(f"{robot_name}/joint_states", JointState)
        self.joint_states_pub = rospy.Publisher("joint_states", JointState, queue_size=10)
        self.goto_home_srv = rospy.ServiceProxy("goto_home", Empty)

        self.service_provider = GazeboServices()

    def reset_robot(self, data_collection_params: Dict):
        pass

    def get_state(self):
        joint_state = self.joint_states_listener.get()
        grippers_res = self.get_grippers_srv(GetDualGripperPointsRequest())
        return {
            'gripper1': ros_numpy.numpify(grippers_res.gripper1),
            'gripper2': ros_numpy.numpify(grippers_res.gripper2),
            'joint_positions': joint_state.position,
            'joint_names': joint_state.name,
        }

    def states_description(self) -> Dict:
        # joints_res = self.joint_states_srv(GetJointStateRequest())
        # FIXME:
        n_joints = 7 + 7 + 14
        return {
            'gripper1': 3,
            'gripper2': 3,
            'joint_positions': n_joints
        }

    def execute_action(self, action: Dict):
        target_left_gripper_point = ros_numpy.msgify(Point, action['gripper1_position'])
        target_right_gripper_point = ros_numpy.msgify(Point, action['gripper2_position'])

        req = GrippersTrajectoryRequest()
        req.speed = 0.1
        left_gripper_points = Points()
        left_gripper_points.points.append(target_left_gripper_point)
        right_gripper_points = Points()
        right_gripper_points.points.append(target_right_gripper_point)
        req.grippers.append(left_gripper_points)
        req.grippers.append(right_gripper_points)
        req.group_name = "both_arms"
        req.tool_names = ["left_tool_placeholder", "right_tool_placeholder"]

        _ = self.action_srv(req)

    def plot_state_rviz(self, state: Dict, label: str, **kwargs):
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.position = state['joint_positions']
        if isinstance(state['joint_names'][0], bytes):
            joint_names = [n.decode("utf-8") for n in state['joint_names']]
        elif isinstance(state['joint_names'][0], str):
            joint_names = [str(n) for n in state['joint_names']]
        else:
            raise NotImplementedError(type(state['joint_names'][0]))
        joint_msg.name = joint_names
        self.joint_states_pub.publish(joint_msg)

    def dynamics_dataset_metadata(self):
        joint_state = self.joint_states_listener.get()
        return {
            'joint_names': joint_state.name
        }

    def simple_name(self):
        return "dual_arm_no_rope"

    def randomize_environment(self, env_rng, objects_params: Dict, data_collection_params: Dict):
        pass

    def get_environment(self, params: Dict, **kwargs):
        return {}
