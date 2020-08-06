from typing import Dict

import numpy as np
import rospy
from std_srvs.srv import Empty, EmptyRequest

import ros_numpy
from link_bot_pycommon.dual_floating_gripper_scenario import DualFloatingGripperRopeScenario
from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, MoveGroupGoal, MoveItErrorCodes
from peter_msgs.srv import GetDualGripperPointsRequest, GetJointStateRequest, GetRopeStateRequest, GetJointState


class DualArmRopeScenario(DualFloatingGripperRopeScenario):

    def __init__(self):
        super().__init__()
        self.joint_states_srv = rospy.ServiceProxy("joint_states", GetJointState)
        self.goto_home_srv = rospy.ServiceProxy("goto_home", Empty)

    def reset_robot(self, data_collection_params: Dict):
        if data_collection_params['scene'] == 'tabletop':
            self.goto_home_srv(EmptyRequest())
        elif data_collection_params['scene'] in ['car', 'car2', 'car-floor']:
            positions = np.array(data_collection_params['reset_robot']['position'])
            names = data_collection_params['reset_robot']['name']

            goal_config_constraint = Constraints()

            for name, position in zip(names, positions):
                joint_constraint = JointConstraint()
                joint_constraint.joint_name = name
                joint_constraint.position = position
                goal_config_constraint.joint_constraints.append(joint_constraint)

            req = MotionPlanRequest()
            req.group_name = 'both_arms'
            req.goal_constraints.append(goal_config_constraint)

            goal = MoveGroupGoal()
            goal.request = req
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
        }

    @staticmethod
    def states_description() -> Dict:
        return {
            'gripper1': 3,
            'gripper2': 3,
            'link_bot': DualArmRopeScenario.n_links * 3,
            'joint_positions': 2 + 7 + 7
        }
