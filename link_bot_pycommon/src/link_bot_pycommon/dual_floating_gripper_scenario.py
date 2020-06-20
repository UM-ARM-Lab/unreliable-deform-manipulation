from typing import Dict, Optional

import numpy as np
import ros_numpy
import tensorflow as tf

import rospy
from geometry_msgs.msg import Point
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from mps_shape_completion_msgs.msg import OccupancyStamped
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest, GetDualGripperPoints
from std_msgs.msg import Empty
from visualization_msgs.msg import MarkerArray


class DualFloatingGripperRopeScenario(Base3DScenario):
    def __init__(self, params: Dict):
        super().__init__(params)
        self.settling_time_seconds = 0.1  # TODO: get this from the data collection params?
        self.action_srv = rospy.ServiceProxy("execute_dual_gripper_action", DualGripperTrajectory)
        self.interrupt = rospy.Publisher("interrupt_trajectory", Empty, queue_size=10)
        self.get_srv = rospy.ServiceProxy("get_dual_gripper_points", GetDualGripperPoints)

        self.env_viz_srv = rospy.Publisher('occupancy', OccupancyStamped, queue_size=10)
        self.state_viz_srv = rospy.Publisher("state_viz", MarkerArray, queue_size=10)
        self.action_viz_srv = rospy.Publisher("action_viz", MarkerArray, queue_size=10)
        self.nudge_rng = np.random.RandomState(0)

    def random_nearby_position_action(self, state: Dict, action_rng: np.random.RandomState, environment):
        while True:
            target_gripper1_pos = Base3DScenario.random_pos(action_rng, environment)
            target_gripper2_pos = Base3DScenario.random_pos(action_rng, environment)
            current_gripper1_pos, current_gripper2_pos = DualFloatingGripperRopeScenario.state_to_gripper_position(state)

            gripper1_displacement = target_gripper1_pos - current_gripper1_pos
            gripper1_displacement = gripper1_displacement / np.linalg.norm(
                gripper1_displacement) * self.params['max_delta_pos'] * action_rng.uniform(0, 1)
            target_gripper1_pos = current_gripper1_pos + gripper1_displacement

            gripper2_displacement = target_gripper2_pos - current_gripper2_pos
            gripper2_displacement = gripper2_displacement / np.linalg.norm(
                gripper2_displacement) * self.params['max_delta_pos'] * action_rng.uniform(0, 1)
            target_gripper2_pos = current_gripper2_pos + gripper2_displacement

            # TODO: this won't prevent overstretching with obstacles...
            distance_between_grippers = np.linalg.norm(target_gripper2_pos - target_gripper1_pos)
            if self.params['min_dist_between_grippers'] < distance_between_grippers < self.params['max_dist_between_grippers']:
                return target_gripper1_pos, target_gripper2_pos

    def execute_action(self, action: Dict):
        target_gripper1_point = ros_numpy.msgify(Point, action['gripper1_position'])
        target_gripper2_point = ros_numpy.msgify(Point, action['gripper2_position'])

        req = DualGripperTrajectoryRequest()
        req.settling_time_seconds = self.settling_time_seconds
        req.gripper1_points.append(target_gripper1_point)
        req.gripper2_points.append(target_gripper2_point)
        _ = self.action_srv(req)

    def nudge(self, state: Dict, environment: Dict):
        nudge_action = self.random_nearby_position_action(state, self.nudge_rng, environment)
        self.execute_action(nudge_action)

    def sample_action(self,
                      environment: Dict,
                      state,
                      last_action: Optional[Dict],
                      params: Dict,
                      action_rng):
        gripper1_position, gripper2_position = self.random_nearby_position_action(state, action_rng, environment)
        return {
            'gripper1_position': gripper1_position,
            'gripper2_position': gripper2_position,
        }

    @staticmethod
    def put_state_local_frame(state: Dict):
        gripper1 = state['gripper1']
        gripper2 = state['gripper2']
        rope = state['link_bot']

        batch_size = gripper1.shape[0]

        gripper1_local = gripper1 - gripper1
        gripper2_local = gripper2 - gripper1

        rope_points = tf.reshape(rope, [batch_size, -1, 3])
        rope_points_local = rope_points - gripper1[:, tf.newaxis]
        rope_local = tf.reshape(rope_points_local, [batch_size, -1])

        return {
            'gripper1': gripper1_local,
            'gripper2': gripper2_local,
            'link_bot': rope_local,
        }

    @staticmethod
    def integrate_dynamics(s_t: Dict, delta_s_t: Dict):
        return {k: s_t[k] + delta_s_t[k] for k in s_t.keys()}

    @staticmethod
    def put_action_local_frame(state: Dict, action: Dict):
        target_gripper1_position = action['gripper1_position']
        target_gripper2_position = action['gripper2_position']

        current_gripper1_point = state['gripper1']
        current_gripper2_point = state['gripper2']

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
    def action_description():
        # should match the keys of the dict return from action_to_dataset_action
        return {
            'gripper1_position': 3,
            'gripper2_position': 3,
        }

    def index_state_time(self, state, t):
        state_t = {}
        for feature_name in ['gripper1', 'gripper2', 'link_bot']:
            state_t[feature_name] = state[feature_name][:, t]
        return state_t

    def index_action_time(self, action, t):
        action_t = {}
        for feature_name in ['gripper1_delta', 'gripper2_delta']:
            if t < action[feature_name].shape[1]:
                action_t[feature_name] = action[feature_name][:, t]
            else:
                action_t[feature_name] = action[feature_name][:, t - 1]
        return action_t

    def __repr__(self):
        return "DualFloatingGripperRope"

    def simple_name(self):
        return "dual_floating_gripper_rope"
