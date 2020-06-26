from copy import deepcopy
from typing import Dict

import numpy as np
import ros_numpy
import tensorflow as tf

import rospy
from geometry_msgs.msg import Point
from link_bot_data.link_bot_dataset_utils import add_predicted
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest, GetDualGripperPoints, WorldControlRequest, \
    WorldControl, SetRopeState, SetRopeStateRequest, SetDualGripperPoints, GetRopeState, GetRopeStateRequest, \
    GetDualGripperPointsRequest
from std_msgs.msg import Empty


class DualFloatingGripperRopeScenario(Base3DScenario):
    def __init__(self, params: Dict):
        super().__init__(params)
        self.last_state = None
        self.last_action = None
        self.action_srv = rospy.ServiceProxy("execute_dual_gripper_action", DualGripperTrajectory)
        self.interrupt = rospy.Publisher("interrupt_trajectory", Empty, queue_size=10)
        self.get_grippers_srv = rospy.ServiceProxy("get_dual_gripper_points", GetDualGripperPoints)
        self.set_rope_srv = rospy.ServiceProxy("set_rope_state", SetRopeState)
        self.get_rope_srv = rospy.ServiceProxy("get_rope_state", GetRopeState)
        self.set_grippers_srv = rospy.ServiceProxy("set_dual_gripper_points", SetDualGripperPoints)
        self.world_control_srv = rospy.ServiceProxy("world_control", WorldControl)

        self.nudge_rng = np.random.RandomState(0)

    def sample_action(self,
                      environment: Dict,
                      state,
                      params: Dict,
                      action_rng):
        while True:
            # move in the same direction as the previous action with 80% probability
            if self.last_action is not None and action_rng.uniform(0, 1) < 0.8:
                last_delta_gripper_1 = state['gripper1'] - self.last_state['gripper1']
                last_delta_gripper_2 = state['gripper2'] - self.last_state['gripper2']
                gripper1_position = state['gripper1'] + last_delta_gripper_1
                gripper2_position = state['gripper2'] + last_delta_gripper_2
            else:
                gripper1_position, gripper2_position = self.random_nearby_position_action(
                    state, action_rng, environment)

            out_of_bounds = self.is_out_of_bounds(gripper1_position) or self.is_out_of_bounds(gripper2_position)
            min_d = self.params['min_distance_between_grippers']
            gripper_collision = np.linalg.norm(gripper2_position - gripper1_position) < min_d
            if not out_of_bounds and not gripper_collision:
                action = {
                    'gripper1_position': gripper1_position,
                    'gripper2_position': gripper2_position,
                }
                self.last_state = deepcopy(state)
                self.last_action = deepcopy(action)
                return action

    def is_out_of_bounds(self, p):
        x, y, z = p
        extent = self.params['action_sample_extent']
        x_min, x_max, y_min, y_max, z_min, z_max = extent
        return x < x_min or x > x_max \
            or y < y_min or y > y_max \
            or z < z_min or z > z_max

    def random_nearby_position_action(self, state: Dict, action_rng: np.random.RandomState, environment):
        max_d = self.params['max_distance_gripper_can_move']
        target_gripper1_pos = Base3DScenario.random_pos(action_rng, self.params['action_sample_extent'])
        target_gripper2_pos = Base3DScenario.random_pos(action_rng, self.params['action_sample_extent'])
        current_gripper1_pos, current_gripper2_pos = DualFloatingGripperRopeScenario.state_to_gripper_position(state)

        gripper1_displacement = target_gripper1_pos - current_gripper1_pos
        gripper1_displacement = gripper1_displacement / np.linalg.norm(gripper1_displacement)
        gripper1_displacement = gripper1_displacement * max_d * action_rng.uniform(0, 1)
        target_gripper1_pos = current_gripper1_pos + gripper1_displacement

        gripper2_displacement = target_gripper2_pos - current_gripper2_pos
        gripper2_displacement = gripper2_displacement / np.linalg.norm(gripper2_displacement)
        gripper2_displacement = gripper2_displacement * max_d * action_rng.uniform(0, 1)
        target_gripper2_pos = current_gripper2_pos + gripper2_displacement

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

    @staticmethod
    def put_state_local_frame(state: Dict):
        rope = state['link_bot']
        rope_points_shape = rope.shape[:-1].as_list() + [-1, 3]
        rope_points = tf.reshape(rope, rope_points_shape)

        center = tf.reduce_mean(rope_points, axis=-2)

        gripper1_local = state['gripper1'] - center
        gripper2_local = state['gripper2'] - center

        rope_points_local = rope_points - tf.expand_dims(center, axis=-2)
        rope_local = tf.reshape(rope_points_local, rope.shape)

        return {
            'gripper1': gripper1_local,
            'gripper2': gripper2_local,
            'link_bot': rope_local,
        }

    @staticmethod
    def local_environment_center_differentiable(state):
        """
        :param state: Dict of batched states
        :return:
        """
        rope_vector = state['link_bot']
        rope_points = tf.reshape(rope_vector, [rope_vector.shape[0], -1, 3])
        center = tf.reduce_mean(rope_points, axis=1)
        return center

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
        for p in rope_res.positions:
            rope_state_vector.append(p.x)
            rope_state_vector.append(p.y)
            rope_state_vector.append(p.z)

        rope_velocity_vector = []
        for v in rope_res.velocities:
            rope_velocity_vector.append(v.x)
            rope_velocity_vector.append(v.y)
            rope_velocity_vector.append(v.z)

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
            'rope_velocities': np.array(rope_velocity_vector, np.float32),
            'model_pose': model_pose,
            'joint_angles_axis1': np.array(rope_res.joint_angles_axis1, np.float32),
            'joint_angles_axis2': np.array(rope_res.joint_angles_axis2, np.float32),
        }

    @staticmethod
    def states_description() -> Dict:
        # should match the keys of the dict return from action_to_dataset_action
        n_links = 15
        # +2 for joints to the grippers
        n_joints = n_links - 1 + 2
        return {
            'gripper1': 3,
            'gripper2': 3,
            'link_bot': n_links * 3,
            'model_pose': 3 + 4,
            'joint_angles_axis1': 2 * n_joints,
            'joint_angles_axis2': 2 * n_joints,
        }

    @staticmethod
    def actions_description() -> Dict:
        # should match the keys of the dict return from action_to_dataset_action
        return {
            'gripper1_position': 3,
            'gripper2_position': 3,
        }

    @staticmethod
    def index_predicted_state_time(state, t):
        state_t = {}
        for feature_name in ['gripper1', 'gripper2', 'link_bot']:
            state_t[feature_name] = state[add_predicted(feature_name)][:, t]
        return state_t

    @staticmethod
    def index_state_time(state, t):
        state_t = {}
        for feature_name in ['gripper1', 'gripper2', 'link_bot']:
            state_t[feature_name] = state[feature_name][:, t]
        return state_t

    @staticmethod
    def index_action_time(action, t):
        action_t = {}
        for feature_name in ['gripper1_position', 'gripper2_position']:
            if t < action[feature_name].shape[1]:
                action_t[feature_name] = action[feature_name][:, t]
            else:
                action_t[feature_name] = action[feature_name][:, t - 1]
        return action_t

    @staticmethod
    def index_label_time(example: Dict, t: int):
        return example['is_close'][:, t]

    def safety_policy(self, previous_state: Dict, new_state: Dict, environment: Dict):
        gripper1_point = new_state['gripper1']
        # the last link connects to gripper 1 at the moment
        rope_state_vector = new_state['link_bot']
        link_point = np.array([rope_state_vector[-3], rope_state_vector[-2], rope_state_vector[-1]])
        distance_between_gripper1_and_link = np.linalg.norm(gripper1_point - link_point)
        rope_is_overstretched = distance_between_gripper1_and_link > self.params['max_dist_between_gripper_and_link']

        if rope_is_overstretched:
            rospy.logwarn("safety policy reversing last action")
            action = {
                'gripper1_position': previous_state['gripper1'],
                'gripper2_position': previous_state['gripper2'],
            }
            self.execute_action(action)

    def __repr__(self):
        return "DualFloatingGripperRope"

    def simple_name(self):
        return "dual_floating_gripper_rope"
