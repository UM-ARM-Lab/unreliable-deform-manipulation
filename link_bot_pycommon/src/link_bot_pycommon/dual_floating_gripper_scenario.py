from typing import Dict, Optional

import numpy as np
import ros_numpy
import tensorflow as tf

import rospy
from geometry_msgs.msg import Point
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from mps_shape_completion_msgs.msg import OccupancyStamped
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest, GetDualGripperPoints, WorldControlRequest, \
    WorldControl, SetRopeState, SetRopeStateRequest, SetDualGripperPoints, GetRopeState, GetRopeStateRequest, \
    GetDualGripperPointsRequest
from std_msgs.msg import Empty
from visualization_msgs.msg import MarkerArray


class DualFloatingGripperRopeScenario(Base3DScenario):
    def __init__(self, params: Dict):
        super().__init__(params)
        self.action_srv = rospy.ServiceProxy("execute_dual_gripper_action", DualGripperTrajectory)
        self.interrupt = rospy.Publisher("interrupt_trajectory", Empty, queue_size=10)
        self.get_grippers_srv = rospy.ServiceProxy("get_dual_gripper_points", GetDualGripperPoints)
        self.set_rope_srv = rospy.ServiceProxy("set_rope_state", SetRopeState)
        self.get_rope_srv = rospy.ServiceProxy("get_rope_state", GetRopeState)
        self.set_grippers_srv = rospy.ServiceProxy("set_dual_gripper_points", SetDualGripperPoints)
        self.world_control_srv = rospy.ServiceProxy("world_control", WorldControl)

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

    def settle(self):
        req = WorldControlRequest()
        req.seconds = self.params['settling_time']
        self.world_control_srv(req)

    def execute_action(self, action: Dict):
        target_gripper1_point = ros_numpy.msgify(Point, action['gripper1_position'])
        target_gripper2_point = ros_numpy.msgify(Point, action['gripper2_position'])

        req = DualGripperTrajectoryRequest()
        req.settling_time_seconds = self.params['settling_time']
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

    def teleport_to_state(self, state: Dict):
        rope_req = SetRopeStateRequest()
        rope_req.joint_angles_axis1 = state['joint_angles_axis1'].tolist()
        rope_req.joint_angles_axis2 = state['joint_angles_axis2'].tolist()
        rope_req.model_pose.position.x = state['model_pose'][0]
        rope_req.model_pose.position.y = state['model_pose'][1]
        rope_req.model_pose.position.z = state['model_pose'][2]
        rope_req.model_pose.orientation.w = state['model_pose'][3]
        rope_req.model_pose.orientation.x = state['model_pose'][4]
        rope_req.model_pose.orientation.y = state['model_pose'][5]
        rope_req.model_pose.orientation.z = state['model_pose'][6]
        self.set_rope_srv(rope_req)

    def get_state(self):
        grippers_res = self.get_grippers_srv(GetDualGripperPointsRequest())
        rope_res = self.get_rope_srv(GetRopeStateRequest())
        rope_state_vector = []
        for p in rope_res.points:
            rope_state_vector.append(p.x)
            rope_state_vector.append(p.y)
            rope_state_vector.append(p.z)

        model_pose = [
            rope_res.model_pose.position.x,
            rope_res.model_pose.position.y,
            rope_res.model_pose.position.z,
            rope_res.model_pose.orientation.w,
            rope_res.model_pose.orientation.x,
            rope_res.model_pose.orientation.y,
            rope_res.model_pose.orientation.z,
        ]

        return {
            'gripper1': ros_numpy.numpify(grippers_res.gripper1),
            'gripper2': ros_numpy.numpify(grippers_res.gripper2),
            'link_bot': np.array(rope_state_vector, np.float32),
            'model_pose': model_pose,
            'joint_angles_axis1': np.array(rope_res.joint_angles_axis1, np.float32),
            'joint_angles_axis2': np.array(rope_res.joint_angles_axis2, np.float32),
        }

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

    @staticmethod
    def index_action_time(action, t):
        action_t = {}
        for feature_name in ['gripper1_delta', 'gripper2_delta']:
            if t < action[feature_name].shape[1]:
                action_t[feature_name] = action[feature_name][:, t]
            else:
                action_t[feature_name] = action[feature_name][:, t - 1]
        return action_t

    def safety_policy(self, previous_state: Dict, new_state: Dict, environment: Dict):
        gripper1_point = new_state['gripper1']
        # the last link connects to gripper 1 at the moment
        rope_state_vector = new_state['link_bot']
        link_point = np.array([rope_state_vector[-3], rope_state_vector[-2], rope_state_vector[-1]])
        distance_between_gripper1_and_link = np.linalg.norm(gripper1_point - link_point)
        rope_is_overstretched = distance_between_gripper1_and_link

        if rope_is_overstretched:
            action = {
                'gripper1_position': previous_state['gripper1'],
                'gripper2_position': previous_state['gripper2'],
            }
            self.execute_action(action)

    def __repr__(self):
        return "DualFloatingGripperRope"

    def simple_name(self):
        return "dual_floating_gripper_rope"