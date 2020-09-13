from typing import Dict

import numpy as np

import actionlib
import ros_numpy
import rospy
from link_bot_pycommon.dual_floating_gripper_scenario import DualFloatingGripperRopeScenario
from link_bot_pycommon.moveit_utils import make_moveit_action_goal
from moveit_msgs.msg import MoveItErrorCodes, MoveGroupAction
from peter_msgs.srv import GetDualGripperPointsRequest, GetJointStateRequest, GetRopeStateRequest, GetJointState
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty


class DualArmRopeScenario(DualFloatingGripperRopeScenario):

    def __init__(self):
        super().__init__()
        self.joint_states_srv = rospy.ServiceProxy("joint_states", GetJointState)
        self.joint_states_pub = rospy.Publisher("joint_states", JointState, queue_size=10)
        self.goto_home_srv = rospy.ServiceProxy("goto_home", Empty)

    def reset_robot(self, data_collection_params: Dict):
        if data_collection_params['scene'] == 'tabletop':
            moveit_client = actionlib.SimpleActionClient('move_group', MoveGroupAction)
            moveit_client.wait_for_server()
            joint_names = data_collection_params['home']['name']
            joint_positions = data_collection_params['home']['position']
            goal = make_moveit_action_goal(joint_names, joint_positions)
            moveit_client.send_goal(goal)
            moveit_client.wait_for_result()
            result = moveit_client.get_result()
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                print("Error! code " + str(result.error_code.val))

        elif data_collection_params['scene'] in ['car', 'car2', 'car-floor']:
            positions = np.array(data_collection_params['reset_robot']['position'])
            names = data_collection_params['reset_robot']['name']

            goal = make_moveit_action_goal(names, positions)
            self.move_group_client.send_goal(goal)
            self.move_group_client.wait_for_result()
            result = self.move_group_client.get_result()

            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                rospy.logwarn(f"Failed to reset robot. Running hard reset.")
                self.hard_reset()

    def get_state(self):
        grippers_res = self.get_grippers_srv(GetDualGripperPointsRequest())
        joints_res = self.joint_states_srv(GetJointStateRequest())
        while True:
            try:
                rope_res = self.get_rope_srv(GetRopeStateRequest())
                break
            except Exception:
                print("CDCPD failed? Restart it!")
                input("press enter.")

        rope_state_vector = []
        for p in rope_res.positions:
            rope_state_vector.append(p.x)
            rope_state_vector.append(p.y)
            rope_state_vector.append(p.z)

        rope_velocity_vector = []
        for v in rope_res.velocities:
            rope_velocity_vector.append(v.x)
            rope_velocity_vector.append(v.y)
            rope_velocity_vector.append(v.z)

        return {
            'gripper1': ros_numpy.numpify(grippers_res.gripper1),
            'gripper2': ros_numpy.numpify(grippers_res.gripper2),
            'link_bot': np.array(rope_state_vector, np.float32),
            'joint_positions': joints_res.joint_state.position,
            'joint_names': joints_res.joint_state.name,
        }

    def states_description(self) -> Dict:
        # joints_res = self.joint_states_srv(GetJointStateRequest())
        # FIXME:
        n_joints = 7 + 7 + 14
        return {
            'gripper1': 3,
            'gripper2': 3,
            'link_bot': DualArmRopeScenario.n_links * 3,
            'joint_positions': n_joints
        }

    def plot_state_rviz(self, state: Dict, label: str, **kwargs):
        super().plot_state_rviz(state, label, **kwargs)
        # TODO: remove this once we no longer need to use the old datasets
        if 'joint_positions' in state and 'joint_names' in state:
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
        joints_res = self.joint_states_srv(GetJointStateRequest())
        return {
            'joint_names': joints_res.joint_state.name
        }

    def simple_name(self):
        return "dual_arm"
