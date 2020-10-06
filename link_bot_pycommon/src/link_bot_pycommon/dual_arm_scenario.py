from typing import Dict

import ros_numpy

import rospy
from arm_robots.get_moveit_robot import get_moveit_robot
from link_bot_gazebo_python.gazebo_services import GazeboServices
from link_bot_pycommon.dual_floating_gripper_scenario import FloatingRopeScenario
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty


class DualArmScenario(FloatingRopeScenario):

    def __init__(self):
        super().__init__()
        self.service_provider = GazeboServices()
        self.joint_state_viz_pub = rospy.Publisher("joint_states_viz", JointState, queue_size=10)
        self.goto_home_srv = rospy.ServiceProxy("goto_home", Empty)

        self.robot = get_moveit_robot()

    def on_before_data_collection(self, params: Dict):
        # move to init positions
        self.robot.plan_to_joint_config("both_arms", params['reset_joint_config'])
        self.robot.close_left_gripper()
        self.robot.close_right_gripper()

    def reset_robot(self, data_collection_params: Dict):
        pass

    def get_gripper_positions(self):
        left_gripper = self.robot.robot_commander.get_link("left_tool_placeholder")
        left_gripper_position = ros_numpy.numpify(left_gripper.pose().pose.position)
        right_gripper = self.robot.robot_commander.get_link("right_tool_placeholder")
        right_gripper_position = ros_numpy.numpify(right_gripper.pose().pose.position)
        return left_gripper_position, right_gripper_position

    def get_state(self):
        joint_state = self.robot.base_robot.joint_state_listener.get()
        left_gripper_position, right_gripper_position = self.get_gripper_positions()
        return {
            'left_gripper': left_gripper_position,
            'right_gripper': right_gripper_position,
            'joint_positions': joint_state.position,
            'joint_names': joint_state.name,
        }

    def states_description(self) -> Dict:
        n_joints = len(self.robot.robot_commander.get_joint_names())
        return {
            'left_gripper': 3,
            'right_gripper': 3,
            'joint_positions': n_joints
        }

    def execute_action(self, action: Dict):
        left_gripper_points = [action['left_gripper_position']]
        right_gripper_points = [action['right_gripper_position']]
        tool_names = ["left_tool_placeholder", "right_tool_placeholder"]
        grippers = [left_gripper_points, right_gripper_points]
        self.robot.follow_jacobian_to_position("both_arms", tool_names, grippers)
        rospy.sleep(0.5)  # REMOVE ME

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
        self.joint_state_viz_pub.publish(joint_msg)

    def dynamics_dataset_metadata(self):
        joint_state = self.robot.base_robot.joint_state_listener.get()
        return {
            'joint_names': joint_state.name
        }

    def simple_name(self):
        return "dual_arm_no_rope"

    def randomize_environment(self, env_rng, objects_params: Dict, data_collection_params: Dict):
        pass

    def get_environment(self, params: Dict, **kwargs):
        return {}
