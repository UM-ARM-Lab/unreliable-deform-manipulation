from typing import Dict, Optional

import numpy as np
import ros_numpy

import rospy
from geometry_msgs.msg import Point
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from link_bot_pycommon.params import CollectDynamicsParams
from mps_shape_completion_msgs.msg import OccupancyStamped
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest, GetDualGripperPoints, GetDualGripperPointsRequest
from std_msgs.msg import Empty
from visualization_msgs.msg import MarkerArray


class DualFloatingGripperRopeScenario(Base3DScenario):
    def __init__(self):
        super().__init__()
        self.settling_time_seconds = 0.1  # TODO: get this from the data collection params?
        self.action_srv = rospy.ServiceProxy("execute_dual_gripper_action", DualGripperTrajectory)
        self.interrupt = rospy.Publisher("interrupt_trajectory", Empty, queue_size=10)
        self.get_srv = rospy.ServiceProxy("get_dual_gripper_points", GetDualGripperPoints)

        self.env_viz_srv = rospy.Publisher('occupancy', OccupancyStamped, queue_size=10)
        self.state_viz_srv = rospy.Publisher("state_viz", MarkerArray, queue_size=10)
        self.action_viz_srv = rospy.Publisher("action_viz", MarkerArray, queue_size=10)
        self.nudge_rng = np.random.RandomState(0)

    @staticmethod
    def random_nearby_position_action(state: Dict, action_rng: np.random.RandomState, environment, max_delta_pos=0.05):
        while True:
            target_gripper1_pos = Base3DScenario.random_pos(action_rng, environment)
            target_gripper2_pos = Base3DScenario.random_pos(action_rng, environment)
            current_gripper1_pos, current_gripper2_pos = DualFloatingGripperRopeScenario.state_to_gripper_position(state)

            gripper1_displacement = target_gripper1_pos - current_gripper1_pos
            gripper1_displacement = gripper1_displacement / np.linalg.norm(
                gripper1_displacement) * max_delta_pos * action_rng.uniform(0, 1)
            target_gripper1_pos = current_gripper1_pos + gripper1_displacement

            gripper2_displacement = target_gripper2_pos - current_gripper2_pos
            gripper2_displacement = gripper2_displacement / np.linalg.norm(
                gripper2_displacement) * max_delta_pos * action_rng.uniform(0, 1)
            target_gripper2_pos = current_gripper2_pos + gripper2_displacement

            # TODO: this won't prevent overstretching with obstacles...
            distance_between_grippers = np.linalg.norm(target_gripper2_pos - target_gripper1_pos)
            if 0.02 < distance_between_grippers < 0.3:
                return target_gripper1_pos, target_gripper2_pos

    def execute_action(self, action: Dict):
        get_req = GetDualGripperPointsRequest()
        get_res = self.get_srv(get_req)
        current_gripper1_point = get_res.gripper1
        current_gripper2_point = get_res.gripper2

        target_gripper1_point = ros_numpy.msgify(Point, action['gripper1_position'])
        target_gripper2_point = ros_numpy.msgify(Point, action['gripper2_position'])

        req = DualGripperTrajectoryRequest()
        req.settling_time_seconds = self.settling_time_seconds
        req.gripper1_points.append(current_gripper1_point)
        req.gripper1_points.append(target_gripper1_point)
        req.gripper2_points.append(current_gripper2_point)
        req.gripper2_points.append(target_gripper2_point)
        _ = self.action_srv(req)

    def nudge(self, state: Dict, environment: Dict):
        nudge_action = DualFloatingGripperRopeScenario.random_nearby_position_action(state, self.nudge_rng, environment)
        self.execute_action(nudge_action)

    @staticmethod
    def sample_action(environment: Dict,
                      service_provider,
                      state,
                      last_action: Optional[Dict],
                      params: CollectDynamicsParams,
                      action_rng):
        gripper1_position, gripper2_position = DualFloatingGripperRopeScenario.random_nearby_position_action(state,
                                                                                                             action_rng,
                                                                                                             environment)
        return {
            'gripper1_position': gripper1_position,
            'gripper2_position': gripper2_position,
        }

    def action_to_dataset_action(self, state: Dict, random_action: Dict):
        target_gripper1_position = random_action['gripper1_position']
        target_gripper2_position = random_action['gripper2_position']

        get_req = GetDualGripperPointsRequest()
        get_res = self.get_srv(get_req)
        current_gripper1_point = ros_numpy.numpify(get_res.gripper1)
        current_gripper2_point = ros_numpy.numpify(get_res.gripper2)

        gripper1_delta = target_gripper1_position - current_gripper1_point
        gripper2_delta = target_gripper2_position - current_gripper2_point

        return {
            'gripper1_delta': gripper1_delta,
            'gripper2_delta': gripper2_delta,
        }

    @staticmethod
    def state_to_gripper_position(state: Dict):
        gripper_position1 = np.reshape(state['gripper1'], [3])
        gripper_position2 = np.reshape(state['gripper2'], [3])
        return gripper_position1, gripper_position2

    @staticmethod
    def dataset_action_description():
        # should match the keys of the dict return from action_to_dataset_action
        return {
            'gripper1_delta': 3,
            'gripper2_delta': 3,
        }

    def __repr__(self):
        return "DualFloatingGripperRope"
