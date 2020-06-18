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
        self.settling_time_seconds = 0.5  # TODO: get this from the data collection params?
        self.action_srv = rospy.ServiceProxy("execute_dual_gripper_action", DualGripperTrajectory)
        self.interrupt = rospy.Publisher("interrupt_trajectory", Empty)
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
            current_gripper1_pos, current_gripper2_pos = Base3DScenario.state_to_gripper_position(state)

            gripper1_nearby = np.linalg.norm(target_gripper1_pos - current_gripper1_pos) < max_delta_pos
            gripper2_nearby = np.linalg.norm(target_gripper2_pos - current_gripper2_pos) < max_delta_pos
            # TODO: this won't prevent overstretching with obstacles...
            not_overstretched = np.linalg.norm(target_gripper2_pos - target_gripper1_pos) < 0.29
            if gripper1_nearby and gripper2_nearby and not_overstretched:
                return target_gripper1_pos, target_gripper2_pos

    @staticmethod
    def random_delta_position_action(state: Dict, action_rng: np.random.RandomState, environment, max_delta_pos=0.05):
        # sample a random point inside the bounds and generate an action in that direction of some max length
        target_pos = Base3DScenario.random_pos(action_rng, environment)
        current_pos = Base3DScenario.state_to_gripper_position(state)
        delta = target_pos - current_pos
        d = np.linalg.norm(delta)
        v = min(max_delta_pos, d)
        delta = delta / d * v
        return [delta[0], delta[1], delta[2]]

    def execute_action(self, action: Dict):
        get_req = GetDualGripperPointsRequest()
        get_res = self.get_srv(get_req)
        current_gripper1_point = get_res.gripper1
        current_gripper2_point = get_res.gripper2

        target_gripper1_point = ros_numpy.msgify(Point, action['gripper1_position'])
        target_gripper2_point = ros_numpy.msgify(Point, action['gripper1_position'])

        req = DualGripperTrajectoryRequest()
        req.settling_time_seconds = self.settling_time_seconds
        req.gripper1_points.append(current_gripper1_point)
        req.gripper1_points.append(target_gripper1_point)
        req.gripper2_points.append(current_gripper2_point)
        req.gripper2_points.append(target_gripper2_point)
        _ = self.action_srv(req)

    def nudge(self, state: Dict, environment: Dict):
        nudge_action = Base3DScenario.random_nearby_position_action(state, self.nudge_rng, environment)
        self.execute_action(nudge_action)

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
            gripper1_position = Base3DScenario.random_nearby_position_action(state, action_rng, environment)
            gripper2_position = Base3DScenario.random_nearby_position_action(state, action_rng, environment)
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

        gripper1_delta = current_gripper1_point - target_gripper1_position
        gripper2_delta = current_gripper2_point - target_gripper2_position

        return {
            'gripper1_delta', gripper1_delta,
            'gripper2_delta', gripper2_delta,
        }

    @staticmethod
    def state_to_gripper_position(state: Dict):
        gripper_position1 = np.reshape(state['gripper1'], [3])
        gripper_position2 = np.reshape(state['gripper2'], [3])
        return gripper_position1, gripper_position2

    @staticmethod
    def dataset_action_keys():
        # should match the keys of the dict return from action_to_dataset_action
        return ['grippe1_delta', 'gripper2_delta']

    def __repr__(self):
        return "DualFloatingGripperRope"
