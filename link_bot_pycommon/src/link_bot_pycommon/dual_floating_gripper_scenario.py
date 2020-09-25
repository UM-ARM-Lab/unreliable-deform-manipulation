from typing import Dict, Optional, List

import numpy as np
import ompl.base as ob
import ompl.control as oc
import tensorflow as tf
from matplotlib import colors

import actionlib
import ros_numpy
import rospy
from geometry_msgs.msg import Point
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_data.visualization import rviz_arrow
from link_bot_pycommon import grid_utils
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from link_bot_pycommon.collision_checking import inflate_tf_3d
from link_bot_pycommon.grid_utils import extent_to_env_size, extent_to_center, extent_array_to_bbox
from link_bot_pycommon.pycommon import default_if_none, directions_3d
from moonshine.base_learned_dynamics_model import dynamics_loss_function, dynamics_points_metrics_function
from moonshine.moonshine_utils import numpify
from moveit_msgs.msg import MoveGroupAction
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest, GetDualGripperPoints, SetRopeState, \
    SetRopeStateRequest, GetRopeState, GetRopeStateRequest, \
    GetDualGripperPointsRequest
from std_srvs.srv import Empty, EmptyRequest, SetBool, SetBoolRequest
from tf import transformations
from visualization_msgs.msg import MarkerArray, Marker


def make_box_marker_from_extents(extent):
    m = Marker()
    ysize, xsize, zsize = extent_to_env_size(extent)
    xcenter, ycenter, zcenter = extent_to_center(extent)
    m.scale.x = xsize
    m.scale.y = ysize
    m.scale.z = zsize
    m.action = Marker.ADD
    m.type = Marker.CUBE
    m.pose.position.x = xcenter
    m.pose.position.y = ycenter
    m.pose.position.z = zcenter
    m.pose.orientation.w = 1
    return m


def sample_rope_and_grippers(rng, g1, g2, p, n_links, kd):
    g1 = g1 + rng.uniform(-0.05, 0.05, 3)
    g2 = g2 + rng.uniform(-0.05, 0.05, 3)
    p = p + rng.uniform(-0.05, 0.05, 3)
    n_exclude = 5
    k = rng.randint(n_exclude, n_links + 1 - n_exclude)
    rope = [g2]
    for i in range(1, k - 1):
        new_p = (p - g2) * (i / (k - 1))
        noise = rng.uniform([-kd, -kd, -kd], [kd, kd, kd], 3)
        new_p = g2 + new_p + noise
        rope.append(new_p)
    rope.append(p)
    for i in range(1, n_links - k + 1):
        new_p = (g1 - p) * i / (n_links - k)
        noise = rng.uniform([-kd, -kd, -kd], [kd, kd, kd], 3)
        new_p = p + new_p + noise
        rope.append(new_p)
    rope = np.array(rope)
    return rope


def sample_rope_grippers(rng, g1, g2, n_links):
    rope = [g2 + rng.uniform(-0.01, 0.01, 3)]
    for _ in range(n_links - 2):
        xmin = min(g1[0], g2[0]) - 0.1
        ymin = min(g1[1], g2[1]) - 0.1
        zmin = min(g1[2], g2[2]) - 0.4
        xmax = max(g1[0], g2[0]) + 0.1
        ymax = max(g1[1], g2[1]) + 0.1
        zmax = max(g1[2], g2[2]) + 0.01
        p = rng.uniform([xmin, ymin, zmin], [xmax, ymax, zmax])
        rope.append(p)
    rope.append(g1 + rng.uniform(-0.01, 0.01, 3))
    return np.array(rope)


def sample_rope(rng, p, n_links, kd: float):
    p = np.array(p, dtype=np.float32)
    n_exclude = 5
    k = rng.randint(n_exclude, n_links + 1 - n_exclude)
    # the kth point of the rope is put at the point p
    rope = [p]
    previous_point = np.copy(p)
    for i in range(0, k):
        noise = rng.uniform([-kd, -kd, -kd / 2], [kd, kd, kd * 1.2], 3)
        previous_point = previous_point + noise
        rope.insert(0, previous_point)
    next_point = np.copy(p)
    for i in range(k, n_links):
        noise = rng.uniform([-kd, -kd, -kd / 2], [kd, kd, kd * 1.2], 3)
        next_point = next_point + noise
        rope.append(next_point)
    rope = np.array(rope)
    return rope


class DualFloatingGripperRopeScenario(Base3DScenario):
    n_links = 25

    def __init__(self):
        super().__init__()
        self.last_action = None
        self.action_srv = rospy.ServiceProxy("execute_dual_gripper_action", DualGripperTrajectory)
        self.grasping_rope_srv = rospy.ServiceProxy("set_grasping_rope", SetBool)
        self.get_grippers_srv = rospy.ServiceProxy("get_dual_gripper_points", GetDualGripperPoints)
        self.get_rope_srv = rospy.ServiceProxy("get_rope_state", GetRopeState)
        self.set_rope_state_srv = rospy.ServiceProxy("set_rope_state", SetRopeState)
        self.reset_srv = rospy.ServiceProxy("gazebo/reset_simulation", Empty)
        self.gripper1_bbox_pub = rospy.Publisher('gripper1_bbox_pub', BoundingBox, queue_size=10, latch=True)
        self.gripper2_bbox_pub = rospy.Publisher('gripper2_bbox_pub', BoundingBox, queue_size=10, latch=True)

        self.max_action_attempts = 500

        self.robot_reset_rng = np.random.RandomState(0)

        self.move_group_client = None

    def get_environment(self, params: Dict, **kwargs):
        return {}

    def hard_reset(self):
        self.reset_srv(EmptyRequest())

    def randomization_initialization(self):
        self.move_group_client = actionlib.SimpleActionClient('move_group', MoveGroupAction)

    def reset_rope(self, data_collection_params: Dict):
        reset = SetRopeStateRequest()

        # TODO: rename this to rope endpoints reset positions or something
        reset.gripper1.x = numpify(data_collection_params['left_gripper_reset_position'][0])
        reset.gripper1.y = numpify(data_collection_params['left_gripper_reset_position'][1])
        reset.gripper1.z = numpify(data_collection_params['left_gripper_reset_position'][2])
        reset.gripper2.x = numpify(data_collection_params['right_gripper_reset_position'][0])
        reset.gripper2.y = numpify(data_collection_params['right_gripper_reset_position'][1])
        reset.gripper2.z = numpify(data_collection_params['right_gripper_reset_position'][2])

        self.set_rope_state_srv(reset)

    def reset_robot(self, data_collection_params: Dict):
        pass

    def batch_stateless_sample_action(self,
                                      environment: Dict,
                                      state: Dict,
                                      batch_size: int,
                                      n_action_samples: int,
                                      n_actions: int,
                                      data_collection_params: Dict,
                                      action_params: Dict,
                                      action_rng: np.random.RandomState):
        del action_rng  # unused, we used tf here
        # Sample a new random action
        pitch_1 = tf.random.uniform([batch_size, n_action_samples, n_actions], -np.pi, np.pi)
        pitch_2 = tf.random.uniform([batch_size, n_action_samples, n_actions], -np.pi, np.pi)
        yaw_1 = tf.random.uniform([batch_size, n_action_samples, n_actions], -np.pi, np.pi)
        yaw_2 = tf.random.uniform([batch_size, n_action_samples, n_actions], -np.pi, np.pi)
        max_d = action_params['max_distance_gripper_can_move']

        displacement1 = tf.random.uniform([batch_size, n_action_samples, n_actions], 0, max_d)
        displacement2 = tf.random.uniform([batch_size, n_action_samples, n_actions], 0, max_d)

        random_directions_1 = directions_3d(pitch_1, yaw_1)
        gripper1_delta_position = random_directions_1 * displacement1[:, :, :, tf.newaxis]

        random_directions_2 = directions_3d(pitch_2, yaw_2)
        gripper2_delta_position = random_directions_2 * displacement2[:, :, :, tf.newaxis]

        # Apply delta
        gripper1_position = state['gripper1'][:, tf.newaxis, tf.newaxis] + gripper1_delta_position
        gripper2_position = state['gripper2'][:, tf.newaxis, tf.newaxis] + gripper2_delta_position

        actions = {
            'gripper1_position': gripper1_position,
            'gripper2_position': gripper2_position,
        }
        return actions

    def sample_action(self,
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state,
                      data_collection_params: Dict,
                      action_params: Dict,
                      no_repeat: Optional[bool] = False,
                      ):
        action = None
        for _ in range(self.max_action_attempts):
            # move in the same direction as the previous action with some probability
            repeat_probability = data_collection_params['repeat_delta_gripper_motion_probability']
            if not no_repeat and self.last_action is not None and action_rng.uniform(0, 1) < repeat_probability:
                gripper1_delta_position = self.last_action['gripper1_delta_position']
                gripper2_delta_position = self.last_action['gripper2_delta_position']
            else:
                # Sample a new random action
                pitch_1 = action_rng.uniform(-np.pi, np.pi)
                pitch_2 = action_rng.uniform(-np.pi, np.pi)
                yaw_1 = action_rng.uniform(-np.pi, np.pi)
                yaw_2 = action_rng.uniform(-np.pi, np.pi)
                displacement1 = action_rng.uniform(0, action_params['max_distance_gripper_can_move'])
                displacement2 = action_rng.uniform(0, action_params['max_distance_gripper_can_move'])

                rotation_matrix_1 = transformations.euler_matrix(0, pitch_1, yaw_1)
                gripper1_delta_position_homo = rotation_matrix_1 @ np.array([1, 0, 0, 1]) * displacement1
                gripper1_delta_position = gripper1_delta_position_homo[:3]

                rotation_matrix_2 = transformations.euler_matrix(0, pitch_2, yaw_2)
                gripper2_delta_position_homo = rotation_matrix_2 @ np.array([1, 0, 0, 1]) * displacement2
                gripper2_delta_position = gripper2_delta_position_homo[:3]

            # Apply delta and check for out of bounds
            gripper1_position = state['gripper1'] + gripper1_delta_position
            gripper2_position = state['gripper2'] + gripper2_delta_position

            action = {
                'gripper1_position': gripper1_position,
                'gripper2_position': gripper2_position,
                'gripper1_delta_position': gripper1_delta_position,
                'gripper2_delta_position': gripper2_delta_position,
            }
            out_of_bounds = DualFloatingGripperRopeScenario.grippers_out_of_bounds(gripper1_position,
                                                                                   gripper2_position,
                                                                                   data_collection_params)

            max_gripper_d = default_if_none(data_collection_params['max_distance_between_grippers'], 1000)
            too_far = np.linalg.norm(gripper1_position - gripper2_position) > max_gripper_d

            if not out_of_bounds and not too_far:
                self.last_action = action
                return action

        rospy.logwarn("Could not find a valid action, executing an invalid one")
        return action

    @staticmethod
    def grippers_out_of_bounds(gripper1, gripper2, data_collection_params: Dict):
        gripper1_extent = data_collection_params['gripper1_action_sample_extent']
        gripper2_extent = data_collection_params['gripper2_action_sample_extent']
        return DualFloatingGripperRopeScenario.is_out_of_bounds(gripper1, gripper1_extent) \
               or DualFloatingGripperRopeScenario.is_out_of_bounds(gripper2, gripper2_extent)

    @staticmethod
    def is_out_of_bounds(p, extent):
        x, y, z = p
        x_min, x_max, y_min, y_max, z_min, z_max = extent
        return x < x_min or x > x_max \
               or y < y_min or y > y_max \
               or z < z_min or z > z_max

    @staticmethod
    def interpolate(start_state, end_state, step_size=0.05):
        gripper1_start = np.array(start_state['gripper1'])
        gripper1_end = np.array(end_state['gripper1'])

        gripper2_start = np.array(start_state['gripper2'])
        gripper2_end = np.array(end_state['gripper2'])

        gripper1_delta = gripper1_end - gripper1_start
        gripper2_delta = gripper2_end - gripper2_start

        gripper1_steps = np.round(np.linalg.norm(gripper1_delta) / step_size).astype(np.int64)
        gripper2_steps = np.round(np.linalg.norm(gripper2_delta) / step_size).astype(np.int64)
        steps = max(gripper1_steps, gripper2_steps)

        interpolated_actions = []
        for t in np.linspace(step_size, 1, steps):
            gripper1_i = gripper1_start + gripper1_delta * t
            gripper2_i = gripper2_start + gripper2_delta * t
            action = {
                'gripper1_position': gripper1_i,
                'gripper2_position': gripper2_i,
            }
            interpolated_actions.append(action)

        return interpolated_actions

    @staticmethod
    def robot_name():
        return "kinematic_rope"

    def initial_obstacle_poses_with_noise(self, env_rng: np.random.RandomState, obstacles: List):
        object_poses = {}
        for obj, pose in self.start_object_poses.items():
            noisy_position = [pose.pose.position.x + env_rng.uniform(-0.05, 0.05),
                              pose.pose.position.y + env_rng.uniform(-0.05, 0.05),
                              pose.pose.position.z + env_rng.uniform(-0.05, 0.05)]

            object_poses[obj] = (noisy_position, ros_numpy.numpify(pose.pose.orientation))
        return object_poses

    def randomize_environment(self, env_rng, objects_params: Dict, data_collection_params: Dict):
        pass

    def execute_action(self, action: Dict):
        target_gripper1_point = ros_numpy.msgify(Point, action['gripper1_position'])

        target_gripper2_point = ros_numpy.msgify(Point, action['gripper2_position'])

        req = DualGripperTrajectoryRequest()
        req.settling_time_seconds = 0.03
        req.gripper1_points.append(target_gripper1_point)
        req.gripper2_points.append(target_gripper2_point)

        while True:
            try:
                _ = self.action_srv(req)
                break
            except Exception:
                input("Did you forget to start the shim?")
                pass

    @staticmethod
    def put_state_in_robot_frame(state: Dict):
        rope = state['link_bot']
        rope_points_shape = rope.shape[:-1].as_list() + [-1, 3]
        rope_points = tf.reshape(rope, rope_points_shape)

        # This assumes robot is at 0 0 0
        robot_position = tf.constant([[0, 0, 0]], tf.float32)
        gripper1_robot = state['gripper1']
        gripper2_robot = state['gripper2']

        rope_points_robot = rope_points - tf.expand_dims(robot_position, axis=-2)
        rope_robot = tf.reshape(rope_points_robot, rope.shape)

        return {
            'gripper1': gripper1_robot,
            'gripper2': gripper2_robot,
            'link_bot': rope_robot,
        }

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
        rope_vector = state['link_bot']
        rope_points = tf.reshape(rope_vector, [rope_vector.shape[0], -1, 3])
        center = tf.reduce_mean(rope_points, axis=1)
        return center

    @staticmethod
    def apply_local_action_at_state(state, local_action):
        return {
            'gripper1_position': state['gripper1'] + local_action['gripper1_delta'],
            'gripper2_position': state['gripper2'] + local_action['gripper2_delta']
        }

    @staticmethod
    def add_noise(action: Dict, noise_rng: np.random.RandomState):
        gripper1_noise = noise_rng.normal(scale=0.01, size=[3])
        gripper2_noise = noise_rng.normal(scale=0.01, size=[3])
        return {
            'gripper1_position': action['gripper1_position'] + gripper1_noise,
            'gripper2_position': action['gripper2_position'] + gripper2_noise
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

    def get_state(self):
        grippers_res = self.get_grippers_srv(GetDualGripperPointsRequest())
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
        }

    @staticmethod
    def states_description() -> Dict:
        return {
            'gripper1': 3,
            'gripper2': 3,
            'link_bot': DualFloatingGripperRopeScenario.n_links * 3,
        }

    @staticmethod
    def actions_description() -> Dict:
        # should match the keys of the dict return from action_to_dataset_action
        return {
            'gripper1_position': 3,
            'gripper2_position': 3,
        }

    @staticmethod
    def state_to_points_for_cc(state: Dict):
        return state['link_bot'].reshape(-1, 3)

    def __repr__(self):
        return "DualFloatingGripperRope"

    def simple_name(self):
        return "dual_floating"

    @staticmethod
    def numpy_to_ompl_state1(state_np: Dict, state_out: ob.CompoundState):
        for i in range(3):
            state_out[0][i] = np.float64(state_np['gripper1'][i])
        for i in range(3):
            state_out[1][i] = np.float64(state_np['gripper2'][i])
        for i in range(DualFloatingGripperRopeScenario.n_links * 3):
            state_out[2][i] = np.float64(state_np['link_bot'][i])
        state_out[3][0] = np.float64(state_np['stdev'][0])
        state_out[4][0] = np.float64(state_np['num_diverged'][0])

    @staticmethod
    def numpy_to_ompl_state(state_np: Dict, state_out: ob.CompoundState):
        rope_points = np.reshape(state_np['link_bot'], [-1, 3])
        for i in range(3):
            state_out[0][i] = np.float64(state_np['gripper1'][i])
        for i in range(3):
            state_out[1][i] = np.float64(state_np['gripper2'][i])
        for j in range(DualFloatingGripperRopeScenario.n_links):
            for i in range(3):
                state_out[2 + j][i] = np.float64(rope_points[j][i])
        state_out[DualFloatingGripperRopeScenario.n_links + 2][0] = np.float64(state_np['stdev'][0])
        state_out[DualFloatingGripperRopeScenario.n_links + 3][0] = np.float64(state_np['num_diverged'][0])

    @staticmethod
    def ompl_state_to_numpy1(ompl_state: ob.CompoundState):
        gripper1 = np.array([ompl_state[0][0], ompl_state[0][1], ompl_state[0][2]])
        gripper2 = np.array([ompl_state[1][0], ompl_state[1][1], ompl_state[1][2]])
        rope = []
        for i in range(DualFloatingGripperRopeScenario.n_links):
            rope.append(ompl_state[2][3 * i + 0])
            rope.append(ompl_state[2][3 * i + 1])
            rope.append(ompl_state[2][3 * i + 2])
        rope = np.array(rope)
        return {
            'gripper1': gripper1,
            'gripper2': gripper2,
            'link_bot': rope,
            'stdev': np.array([ompl_state[3][0]]),
            'num_diverged': np.array([ompl_state[4][0]]),
        }

    @staticmethod
    def ompl_state_to_numpy(ompl_state: ob.CompoundState):
        gripper1 = np.array([ompl_state[0][0], ompl_state[0][1], ompl_state[0][2]])
        gripper2 = np.array([ompl_state[1][0], ompl_state[1][1], ompl_state[1][2]])
        rope = []
        for i in range(DualFloatingGripperRopeScenario.n_links):
            rope.append(ompl_state[2 + i][0])
            rope.append(ompl_state[2 + i][1])
            rope.append(ompl_state[2 + i][2])
        rope = np.array(rope)
        return {
            'gripper1': gripper1,
            'gripper2': gripper2,
            'link_bot': rope,
            'stdev': np.array([ompl_state[DualFloatingGripperRopeScenario.n_links + 2][0]]),
            'num_diverged': np.array([ompl_state[DualFloatingGripperRopeScenario.n_links + 3][0]]),
        }

    @staticmethod
    def ompl_control_to_numpy(ompl_state: ob.CompoundState, ompl_control: oc.CompoundControl):
        state_np = DualFloatingGripperRopeScenario.ompl_state_to_numpy(ompl_state)
        current_gripper1_position = state_np['gripper1']
        current_gripper2_position = state_np['gripper2']

        rotation_matrix_1 = transformations.euler_matrix(0, ompl_control[0][0], ompl_control[0][1])
        gripper1_delta_position_homo = rotation_matrix_1 @ np.array([1, 0, 0, 1]) * ompl_control[0][2]
        gripper1_delta_position = gripper1_delta_position_homo[:3]

        rotation_matrix_2 = transformations.euler_matrix(0, ompl_control[1][0], ompl_control[1][1])
        gripper2_delta_position_homo = rotation_matrix_2 @ np.array([1, 0, 0, 1]) * ompl_control[1][2]
        gripper2_delta_position = gripper2_delta_position_homo[:3]

        target_gripper1_position = current_gripper1_position + gripper1_delta_position
        target_gripper2_position = current_gripper2_position + gripper2_delta_position
        return {
            'gripper1_position': target_gripper1_position,
            'gripper2_position': target_gripper2_position,
        }

    @staticmethod
    def sample_gripper_goal(environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        env_inflated = inflate_tf_3d(env=environment['env'],
                                     radius_m=planner_params['goal_threshold'], res=environment['res'])
        goal_extent = planner_params['goal_extent']

        while True:
            extent = np.array(goal_extent).reshape(3, 2)
            gripper1 = rng.uniform(extent[:, 0], extent[:, 1])
            gripper2 = rng.uniform(extent[:, 0], extent[:, 1])
            goal = {
                'gripper1': gripper1,
                'gripper2': gripper2,
            }
            row1, col1, channel1 = grid_utils.point_to_idx_3d_in_env(
                gripper1[0], gripper1[1], gripper1[2], environment)
            row2, col2, channel2 = grid_utils.point_to_idx_3d_in_env(
                gripper2[0], gripper2[1], gripper2[2], environment)
            collision1 = env_inflated[row1, col1, channel1] > 0.5
            collision2 = env_inflated[row2, col2, channel2] > 0.5
            if not collision1 and not collision2:
                return goal

    def sample_goal(self, environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        if planner_params['goal_type'] == 'midpoint':
            return self.sample_midpoint_goal(environment, rng, planner_params)
        else:
            raise NotImplementedError(planner_params['goal_type'])

    @staticmethod
    def distance_to_gripper_goal(state: Dict, goal: Dict):
        gripper1 = state['gripper1']
        gripper2 = state['gripper2']
        distance1 = np.linalg.norm(goal['gripper1'] - gripper1)
        distance2 = np.linalg.norm(goal['gripper2'] - gripper2)
        return max(distance1, distance2)

    def sample_midpoint_goal(self, environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        env_inflated = inflate_tf_3d(env=environment['env'],
                                     radius_m=planner_params['goal_threshold'], res=environment['res'])
        # from copy import deepcopy
        # environment_ = deepcopy(environment)
        # environment_['env'] = env_inflated
        # self.plot_environment_rviz(environment_)
        goal_extent = planner_params['goal_extent']

        while True:
            extent = np.array(goal_extent).reshape(3, 2)
            p = rng.uniform(extent[:, 0], extent[:, 1])
            goal = {'midpoint': p}
            row, col, channel = grid_utils.point_to_idx_3d_in_env(p[0], p[1], p[2], environment)
            collision = env_inflated[row, col, channel] > 0.5
            if not collision:
                return goal

    @staticmethod
    def distance_grippers_and_any_point_goal(state: Dict, goal: Dict):
        rope_points = np.reshape(state['link_bot'], [-1, 3])
        # well ok not _any_ node, but ones near the middle
        n_from_ends = 5
        distances = np.linalg.norm(np.expand_dims(goal['point'], axis=0) -
                                   rope_points, axis=1)[n_from_ends:-n_from_ends]
        rope_distance = np.min(distances)

        gripper1 = state['gripper1']
        gripper2 = state['gripper2']
        distance1 = np.linalg.norm(goal['gripper1'] - gripper1)
        distance2 = np.linalg.norm(goal['gripper2'] - gripper2)
        return max(max(distance1, distance2), rope_distance)

    @staticmethod
    def distance_to_any_point_goal(state: Dict, goal: Dict):
        rope_points = np.reshape(state['link_bot'], [-1, 3])
        # well ok not _any_ node, but ones near the middle
        n_from_ends = 7
        distances = np.linalg.norm(np.expand_dims(goal['point'], axis=0) -
                                   rope_points, axis=1)[n_from_ends:-n_from_ends]
        min_distance = np.min(distances)
        return min_distance

    @staticmethod
    def distance_to_midpoint_goal(state: Dict, goal: Dict):
        rope_points = np.reshape(state['link_bot'], [-1, 3])
        rope_midpoint = rope_points[int(DualFloatingGripperRopeScenario.n_links / 2)]
        distance = np.linalg.norm(goal['midpoint'] - rope_midpoint)
        return distance

    @staticmethod
    def full_distance_tf(s1: Dict, s2: Dict):
        """ the same as the distance metric used in planning """
        distance = tf.linalg.norm(s1['link_bot'] - s2['link_bot'], axis=-1)
        return distance

    def batch_full_distance(self, s1: Dict, s2: Dict):
        return np.linalg.norm(s1['link_bot'] - s2['link_bot'], axis=1)

    @staticmethod
    def compute_label(actual: Dict, predicted: Dict, labeling_params: Dict):
        # NOTE: this should be using the same distance metric as the planning, which should also be the same as the labeling
        # done when making the classifier dataset
        actual_rope = np.array(actual["link_bot"])
        predicted_rope = np.array(predicted["link_bot"])
        model_error = np.linalg.norm(actual_rope - predicted_rope)
        threshold = labeling_params['threshold']
        is_close = model_error < threshold
        return is_close

    def distance_to_goal(self, state, goal):
        if 'type' not in goal or goal['type'] == 'midpoint':
            return self.distance_to_midpoint_goal(state, goal)
        elif goal['type'] == 'any_point':
            return self.distance_to_any_point_goal(state, goal)
        elif goal['type'] == 'grippers':
            return self.distance_to_gripper_goal(state, goal)
        elif goal['type'] == 'grippers_and_point':
            return self.distance_grippers_and_any_point_goal(state, goal)
        else:
            raise NotImplementedError()

    def make_goal_region(self, si: oc.SpaceInformation, rng: np.random.RandomState, params: Dict, goal: Dict,
                         plot: bool):
        if 'type' not in goal or goal['type'] == 'midpoint':
            return RopeMidpointGoalRegion(si=si,
                                          scenario=self,
                                          rng=rng,
                                          threshold=params['goal_threshold'],
                                          goal=goal,
                                          plot=plot)
        elif goal['type'] == 'any_point':
            return RopeAnyPointGoalRegion(si=si,
                                          scenario=self,
                                          rng=rng,
                                          threshold=params['goal_threshold'],
                                          goal=goal,
                                          plot=plot)
        elif goal['type'] == 'grippers':
            return DualGripperGoalRegion(si=si,
                                         scenario=self,
                                         rng=rng,
                                         threshold=params['goal_threshold'],
                                         goal=goal,
                                         plot=plot)
        elif goal['type'] == 'grippers_and_point':
            return RopeAndGrippersGoalRegion(si=si,
                                             scenario=self,
                                             rng=rng,
                                             threshold=params['goal_threshold'],
                                             goal=goal,
                                             plot=plot)
        else:
            raise NotImplementedError()

    def make_ompl_state_space(self, planner_params, state_sampler_rng: np.random.RandomState, plot: bool):
        state_space = ob.CompoundStateSpace()

        min_x, max_x, min_y, max_y, min_z, max_z = planner_params['extent']

        gripper1_subspace = ob.RealVectorStateSpace(3)
        gripper1_bounds = ob.RealVectorBounds(3)
        gripper1_bounds.setLow(0, min_x)
        gripper1_bounds.setHigh(0, max_x)
        gripper1_bounds.setLow(1, min_y)
        gripper1_bounds.setHigh(1, max_y)
        gripper1_bounds.setLow(2, min_z)
        gripper1_bounds.setHigh(2, max_z)
        gripper1_subspace.setBounds(gripper1_bounds)
        gripper1_subspace.setName("gripper1")
        state_space.addSubspace(gripper1_subspace, weight=1)

        gripper2_subspace = ob.RealVectorStateSpace(3)
        gripper2_bounds = ob.RealVectorBounds(3)
        gripper2_bounds.setLow(0, min_x)
        gripper2_bounds.setHigh(0, max_x)
        gripper2_bounds.setLow(1, min_y)
        gripper2_bounds.setHigh(1, max_y)
        gripper2_bounds.setLow(2, min_z)
        gripper2_bounds.setHigh(2, max_z)
        gripper2_subspace.setBounds(gripper2_bounds)
        gripper2_subspace.setName("gripper2")
        state_space.addSubspace(gripper2_subspace, weight=1)

        for i in range(DualFloatingGripperRopeScenario.n_links):
            rope_point_subspace = ob.RealVectorStateSpace(3)
            rope_point_bounds = ob.RealVectorBounds(3)
            rope_point_bounds.setLow(0, min_x)
            rope_point_bounds.setHigh(0, max_x)
            rope_point_bounds.setLow(1, min_y)
            rope_point_bounds.setHigh(1, max_y)
            rope_point_bounds.setLow(2, min_z)
            rope_point_bounds.setHigh(2, max_z)
            rope_point_subspace.setBounds(rope_point_bounds)
            rope_point_subspace.setName(f"rope_{i}")
            state_space.addSubspace(rope_point_subspace, weight=1)

        # extra subspace component for the variance, which is necessary to pass information from propagate to constraint checker
        stdev_subspace = ob.RealVectorStateSpace(1)
        stdev_bounds = ob.RealVectorBounds(1)
        stdev_bounds.setLow(-1000)
        stdev_bounds.setHigh(1000)
        stdev_subspace.setBounds(stdev_bounds)
        stdev_subspace.setName("stdev")
        state_space.addSubspace(stdev_subspace, weight=0)

        # extra subspace component for the number of diverged steps
        num_diverged_subspace = ob.RealVectorStateSpace(1)
        num_diverged_bounds = ob.RealVectorBounds(1)
        num_diverged_bounds.setLow(-1000)
        num_diverged_bounds.setHigh(1000)
        num_diverged_subspace.setBounds(num_diverged_bounds)
        num_diverged_subspace.setName("stdev")
        state_space.addSubspace(num_diverged_subspace, weight=0)

        def _state_sampler_allocator(state_space):
            return DualGripperStateSampler(state_space,
                                           scenario=self,
                                           extent=planner_params['state_sampler_extent'],
                                           rng=state_sampler_rng,
                                           plot=plot)

        state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(_state_sampler_allocator))

        return state_space

    def make_ompl_control_space(self, state_space, rng: np.random.RandomState, action_params: Dict):
        control_space = oc.CompoundControlSpace(state_space)

        gripper1_control_space = oc.RealVectorControlSpace(state_space, 3)
        gripper1_control_bounds = ob.RealVectorBounds(3)
        # Pitch
        gripper1_control_bounds.setLow(0, -np.pi)
        gripper1_control_bounds.setHigh(0, np.pi)
        # Yaw
        gripper1_control_bounds.setLow(1, -np.pi)
        gripper1_control_bounds.setHigh(1, np.pi)
        # Displacement
        max_d = action_params['max_distance_gripper_can_move']
        gripper1_control_bounds.setLow(2, 0)
        gripper1_control_bounds.setHigh(2, max_d)
        gripper1_control_space.setBounds(gripper1_control_bounds)
        control_space.addSubspace(gripper1_control_space)

        gripper2_control_space = oc.RealVectorControlSpace(state_space, 3)
        gripper2_control_bounds = ob.RealVectorBounds(3)
        # Pitch
        gripper2_control_bounds.setLow(0, -np.pi)
        gripper2_control_bounds.setHigh(0, np.pi)
        # Yaw
        gripper2_control_bounds.setLow(1, -np.pi)
        gripper2_control_bounds.setHigh(1, np.pi)
        # Displacement
        max_d = action_params['max_distance_gripper_can_move']
        gripper2_control_bounds.setLow(2, 0)
        gripper2_control_bounds.setHigh(2, max_d)

        gripper2_control_space.setBounds(gripper2_control_bounds)
        control_space.addSubspace(gripper2_control_space)

        def _allocator(cs):
            return DualGripperControlSampler(cs, scenario=self, rng=rng, action_params=action_params)

        # I override the sampler here so I can use numpy RNG to make things more deterministic.
        # ompl does not allow resetting of seeds, which causes problems when evaluating multiple
        # planning queries in a row.
        control_space.setControlSamplerAllocator(oc.ControlSamplerAllocator(_allocator))

        return control_space

    def plot_goal_rviz(self, goal: Dict, goal_threshold: float, actually_at_goal: Optional[bool] = None):
        if actually_at_goal:
            r = 0.4
            g = 0.8
            b = 0.4
            a = 0.6
        else:
            r = 0.5
            g = 0.3
            b = 0.8
            a = 0.6

        goal_marker_msg = MarkerArray()

        if 'midpoint' in goal:
            midpoint_marker = Marker()
            midpoint_marker.scale.x = goal_threshold * 2
            midpoint_marker.scale.y = goal_threshold * 2
            midpoint_marker.scale.z = goal_threshold * 2
            midpoint_marker.action = Marker.ADD
            midpoint_marker.type = Marker.SPHERE
            midpoint_marker.header.frame_id = "world"
            midpoint_marker.header.stamp = rospy.Time.now()
            midpoint_marker.ns = 'goal'
            midpoint_marker.id = 0
            midpoint_marker.color.r = r
            midpoint_marker.color.g = g
            midpoint_marker.color.b = b
            midpoint_marker.color.a = a
            midpoint_marker.pose.position.x = goal['midpoint'][0]
            midpoint_marker.pose.position.y = goal['midpoint'][1]
            midpoint_marker.pose.position.z = goal['midpoint'][2]
            midpoint_marker.pose.orientation.w = 1
            goal_marker_msg.markers.append(midpoint_marker)

        if 'point' in goal:
            point_marker = Marker()
            point_marker.scale.x = goal_threshold * 2
            point_marker.scale.y = goal_threshold * 2
            point_marker.scale.z = goal_threshold * 2
            point_marker.action = Marker.ADD
            point_marker.type = Marker.SPHERE
            point_marker.header.frame_id = "world"
            point_marker.header.stamp = rospy.Time.now()
            point_marker.ns = 'goal'
            point_marker.id = 0
            point_marker.color.r = r
            point_marker.color.g = g
            point_marker.color.b = b
            point_marker.color.a = a
            point_marker.pose.position.x = goal['point'][0]
            point_marker.pose.position.y = goal['point'][1]
            point_marker.pose.position.z = goal['point'][2]
            point_marker.pose.orientation.w = 1
            goal_marker_msg.markers.append(point_marker)

        if 'gripper1' in goal:
            gripper1_marker = Marker()
            gripper1_marker.scale.x = goal_threshold * 2
            gripper1_marker.scale.y = goal_threshold * 2
            gripper1_marker.scale.z = goal_threshold * 2
            gripper1_marker.action = Marker.ADD
            gripper1_marker.type = Marker.SPHERE
            gripper1_marker.header.frame_id = "world"
            gripper1_marker.header.stamp = rospy.Time.now()
            gripper1_marker.ns = 'goal'
            gripper1_marker.id = 1
            gripper1_marker.color.r = r
            gripper1_marker.color.g = g
            gripper1_marker.color.b = b
            gripper1_marker.color.a = a
            gripper1_marker.pose.position.x = goal['gripper1'][0]
            gripper1_marker.pose.position.y = goal['gripper1'][1]
            gripper1_marker.pose.position.z = goal['gripper1'][2]
            gripper1_marker.pose.orientation.w = 1
            goal_marker_msg.markers.append(gripper1_marker)

        if 'gripper2' in goal:
            gripper2_marker = Marker()
            gripper2_marker.scale.x = goal_threshold * 2
            gripper2_marker.scale.y = goal_threshold * 2
            gripper2_marker.scale.z = goal_threshold * 2
            gripper2_marker.action = Marker.ADD
            gripper2_marker.type = Marker.SPHERE
            gripper2_marker.header.frame_id = "world"
            gripper2_marker.header.stamp = rospy.Time.now()
            gripper2_marker.ns = 'goal'
            gripper2_marker.id = 2
            gripper2_marker.color.r = r
            gripper2_marker.color.g = g
            gripper2_marker.color.b = b
            gripper2_marker.color.a = a
            gripper2_marker.pose.position.x = goal['gripper2'][0]
            gripper2_marker.pose.position.y = goal['gripper2'][1]
            gripper2_marker.pose.position.z = goal['gripper2'][2]
            gripper2_marker.pose.orientation.w = 1
            goal_marker_msg.markers.append(gripper2_marker)

        self.state_viz_pub.publish(goal_marker_msg)

    def plot_goal_boxes(self, goal: Dict, goal_threshold: float, actually_at_goal: Optional[bool] = None):
        if actually_at_goal:
            r = 0.4
            g = 0.8
            b = 0.4
            a = 0.6
        else:
            r = 0.5
            g = 0.3
            b = 0.8
            a = 0.6

        goal_marker_msg = MarkerArray()

        if 'point_box' in goal:
            point_marker = make_box_marker_from_extents(goal['point_box'])
            point_marker.header.frame_id = "world"
            point_marker.header.stamp = rospy.Time.now()
            point_marker.ns = 'goal'
            point_marker.id = 0
            point_marker.color.r = r
            point_marker.color.g = g
            point_marker.color.b = b
            point_marker.color.a = a
            goal_marker_msg.markers.append(point_marker)

        if 'gripper1_box' in goal:
            gripper1_marker = make_box_marker_from_extents(goal['gripper1_box'])
            gripper1_marker.header.frame_id = "world"
            gripper1_marker.header.stamp = rospy.Time.now()
            gripper1_marker.ns = 'goal'
            gripper1_marker.id = 1
            gripper1_marker.color.r = r
            gripper1_marker.color.g = g
            gripper1_marker.color.b = b
            gripper1_marker.color.a = a
            goal_marker_msg.markers.append(gripper1_marker)

        if 'gripper2_box' in goal:
            gripper2_marker = make_box_marker_from_extents(goal['gripper2_box'])
            gripper2_marker.header.frame_id = "world"
            gripper2_marker.header.stamp = rospy.Time.now()
            gripper2_marker.ns = 'goal'
            gripper2_marker.id = 2
            gripper2_marker.color.r = r
            gripper2_marker.color.g = g
            gripper2_marker.color.b = b
            gripper2_marker.color.a = a
            goal_marker_msg.markers.append(gripper2_marker)

        self.state_viz_pub.publish(goal_marker_msg)

    @staticmethod
    def dynamics_loss_function(dataset_element, predictions):
        return dynamics_loss_function(dataset_element, predictions)

    @staticmethod
    def dynamics_metrics_function(dataset_element, predictions):
        return dynamics_points_metrics_function(dataset_element, predictions)

    def plot_tree_action(self, state: Dict, action: Dict, **kwargs):
        r = kwargs.pop("r", 0.2)
        g = kwargs.pop("g", 0.2)
        b = kwargs.pop("b", 0.8)
        a = kwargs.pop("a", 1.0)
        idx1 = self.tree_action_idx * 2 + 0
        idx2 = self.tree_action_idx * 2 + 1
        self.plot_action_rviz(state, action, label='tree', color=[r, g, b, a], idx1=idx1, idx2=idx2, **kwargs)
        self.tree_action_idx += 1

    def plot_executed_action(self, state: Dict, action: Dict, **kwargs):
        self.plot_action_rviz(state, action, label='executed action', color="#3876EB", idx1=1, idx2=1, **kwargs)

    def plot_state_rviz(self, state: Dict, label: str, **kwargs):
        r, g, b, a = colors.to_rgba(kwargs.get("color", "r"))
        idx = kwargs.get("idx", 0)

        link_bot_points = np.reshape(state['link_bot'], [-1, 3])

        msg = MarkerArray()
        lines = Marker()
        lines.action = Marker.ADD  # create or modify
        lines.type = Marker.LINE_STRIP
        lines.header.frame_id = "world"
        lines.header.stamp = rospy.Time.now()
        lines.ns = label
        lines.id = 6 * idx + 0

        lines.pose.position.x = 0
        lines.pose.position.y = 0
        lines.pose.position.z = 0
        lines.pose.orientation.x = 0
        lines.pose.orientation.y = 0
        lines.pose.orientation.z = 0
        lines.pose.orientation.w = 1

        lines.scale.x = 0.01

        lines.color.r = r
        lines.color.g = g
        lines.color.b = b
        lines.color.a = a

        spheres = Marker()
        spheres.action = Marker.ADD  # create or modify
        spheres.type = Marker.SPHERE_LIST
        spheres.header.frame_id = "world"
        spheres.header.stamp = rospy.Time.now()
        spheres.ns = label
        spheres.id = 6 * idx + 1

        spheres.scale.x = 0.02
        spheres.scale.y = 0.02
        spheres.scale.z = 0.02

        spheres.pose.position.x = 0
        spheres.pose.position.y = 0
        spheres.pose.position.z = 0
        spheres.pose.orientation.x = 0
        spheres.pose.orientation.y = 0
        spheres.pose.orientation.z = 0
        spheres.pose.orientation.w = 1

        spheres.color.r = r
        spheres.color.g = g
        spheres.color.b = b
        spheres.color.a = a

        for i, (x, y, z) in enumerate(link_bot_points):
            point = Point()
            point.x = x
            point.y = y
            point.z = z

            spheres.points.append(point)
            lines.points.append(point)

        gripper1_sphere = Marker()
        gripper1_sphere.action = Marker.ADD  # create or modify
        gripper1_sphere.type = Marker.SPHERE
        gripper1_sphere.header.frame_id = "world"
        gripper1_sphere.header.stamp = rospy.Time.now()
        gripper1_sphere.ns = label
        gripper1_sphere.id = 6 * idx + 2

        gripper1_sphere.scale.x = 0.02
        gripper1_sphere.scale.y = 0.02
        gripper1_sphere.scale.z = 0.02

        gripper1_sphere.pose.position.x = state['gripper1'][0]
        gripper1_sphere.pose.position.y = state['gripper1'][1]
        gripper1_sphere.pose.position.z = state['gripper1'][2]
        gripper1_sphere.pose.orientation.x = 0
        gripper1_sphere.pose.orientation.y = 0
        gripper1_sphere.pose.orientation.z = 0
        gripper1_sphere.pose.orientation.w = 1

        gripper1_sphere.color.r = 0.2
        gripper1_sphere.color.g = 0.2
        gripper1_sphere.color.b = 0.8
        gripper1_sphere.color.a = a

        gripper2_sphere = Marker()
        gripper2_sphere.action = Marker.ADD  # create or modify
        gripper2_sphere.type = Marker.SPHERE
        gripper2_sphere.header.frame_id = "world"
        gripper2_sphere.header.stamp = rospy.Time.now()
        gripper2_sphere.ns = label
        gripper2_sphere.id = 6 * idx + 3

        gripper2_sphere.scale.x = 0.02
        gripper2_sphere.scale.y = 0.02
        gripper2_sphere.scale.z = 0.02

        gripper2_sphere.pose.position.x = state['gripper2'][0]
        gripper2_sphere.pose.position.y = state['gripper2'][1]
        gripper2_sphere.pose.position.z = state['gripper2'][2]
        gripper2_sphere.pose.orientation.x = 0
        gripper2_sphere.pose.orientation.y = 0
        gripper2_sphere.pose.orientation.z = 0
        gripper2_sphere.pose.orientation.w = 1

        gripper2_sphere.color.r = 0.8
        gripper2_sphere.color.g = 0.2
        gripper2_sphere.color.b = 0.2
        gripper2_sphere.color.a = a

        gripper1_text = Marker()
        gripper1_text.action = Marker.ADD  # create or modify
        gripper1_text.type = Marker.TEXT_VIEW_FACING
        gripper1_text.header.frame_id = "world"
        gripper1_text.header.stamp = rospy.Time.now()
        gripper1_text.ns = label
        gripper1_text.id = 6 * idx + 4
        gripper1_text.text = "L"
        gripper1_text.scale.z = 0.015

        gripper1_text.pose.position.x = state['gripper1'][0]
        gripper1_text.pose.position.y = state['gripper1'][1]
        gripper1_text.pose.position.z = state['gripper1'][2] + 0.015
        gripper1_text.pose.orientation.x = 0
        gripper1_text.pose.orientation.y = 0
        gripper1_text.pose.orientation.z = 0
        gripper1_text.pose.orientation.w = 1

        gripper1_text.color.r = 1.0
        gripper1_text.color.g = 1.0
        gripper1_text.color.b = 1.0
        gripper1_text.color.a = 1.0

        midpoint_sphere = Marker()
        midpoint_sphere.action = Marker.ADD  # create or modify
        midpoint_sphere.type = Marker.SPHERE
        midpoint_sphere.header.frame_id = "world"
        midpoint_sphere.header.stamp = rospy.Time.now()
        midpoint_sphere.ns = label
        midpoint_sphere.id = 6 * idx + 5

        midpoint_sphere.scale.x = 0.03
        midpoint_sphere.scale.y = 0.03
        midpoint_sphere.scale.z = 0.03

        rope_midpoint = link_bot_points[int(DualFloatingGripperRopeScenario.n_links / 2)]
        midpoint_sphere.pose.position.x = rope_midpoint[0]
        midpoint_sphere.pose.position.y = rope_midpoint[1]
        midpoint_sphere.pose.position.z = rope_midpoint[2]
        midpoint_sphere.pose.orientation.x = 0
        midpoint_sphere.pose.orientation.y = 0
        midpoint_sphere.pose.orientation.z = 0
        midpoint_sphere.pose.orientation.w = 1

        midpoint_sphere.color.r = r * 0.8
        midpoint_sphere.color.g = g * 0.8
        midpoint_sphere.color.b = b * 0.8
        midpoint_sphere.color.a = a

        msg.markers.append(spheres)
        msg.markers.append(gripper1_sphere)
        msg.markers.append(gripper2_sphere)
        msg.markers.append(gripper1_text)
        msg.markers.append(lines)
        msg.markers.append(midpoint_sphere)
        self.state_viz_pub.publish(msg)

    def plot_action_rviz(self, state: Dict, action: Dict, label: str = 'action', **kwargs):
        state_action = {}
        state_action.update(state)
        state_action.update(action)
        self.plot_action_rviz_internal(state_action, label=label, **kwargs)

    def plot_action_rviz_internal(self, data: Dict, label: str, **kwargs):
        r, g, b, a = colors.to_rgba(kwargs.get("color", "b"))
        s1 = np.reshape(data['gripper1'], [3])
        s2 = np.reshape(data['gripper2'], [3])
        a1 = np.reshape(data['gripper1_position'], [3])
        a2 = np.reshape(data['gripper2_position'], [3])

        idx = kwargs.pop("idx", None)
        if idx is not None:
            idx1 = idx
            idx2 = idx + 100
        else:
            idx1 = kwargs.pop("idx1", 0)
            idx2 = kwargs.pop("idx2", 1)

        msg = MarkerArray()
        msg.markers.append(rviz_arrow(s1, a1, r, g, b, a, idx=idx1, label=label, **kwargs))
        msg.markers.append(rviz_arrow(s2, a2, r, g, b, a, idx=idx2, label=label, **kwargs))

        self.action_viz_pub.publish(msg)


class DualGripperControlSampler(oc.ControlSampler):
    def __init__(self,
                 control_space: oc.CompoundControlSpace,
                 scenario: DualFloatingGripperRopeScenario,
                 rng: np.random.RandomState,
                 action_params: Dict):
        super().__init__(control_space)
        self.scenario = scenario
        self.rng = rng
        self.control_space = control_space
        self.action_params = action_params

    def sampleNext(self, control_out, previous_control, state):
        del previous_control
        del state

        # Pitch
        pitch_1 = self.rng.uniform(-np.pi, np.pi)
        pitch_2 = self.rng.uniform(-np.pi, np.pi)
        # Yaw
        yaw_1 = self.rng.uniform(-np.pi, np.pi)
        yaw_2 = self.rng.uniform(-np.pi, np.pi)
        # Displacement
        displacement1 = self.rng.uniform(0, self.action_params['max_distance_gripper_can_move'])
        displacement2 = self.rng.uniform(0, self.action_params['max_distance_gripper_can_move'])

        control_out[0][0] = pitch_1
        control_out[0][1] = yaw_1
        control_out[0][2] = displacement1

        control_out[1][0] = pitch_2
        control_out[1][1] = yaw_2
        control_out[1][2] = displacement2

    def sampleStepCount(self, min_steps, max_steps):
        step_count = self.rng.randint(min_steps, max_steps)
        return step_count


class DualGripperStateSampler(ob.CompoundStateSampler):

    def __init__(self,
                 state_space,
                 scenario: DualFloatingGripperRopeScenario,
                 extent,
                 rng: np.random.RandomState,
                 plot: bool):
        super().__init__(state_space)
        self.state_space = state_space
        self.scenario = scenario
        self.extent = np.array(extent).reshape(3, 2)
        self.rng = rng
        self.plot = plot

    def sample_point_for_R3_subspace(self, subspace, subspace_state_out):
        bounds = subspace.getBounds()
        min_x = bounds.low[0]
        min_y = bounds.low[1]
        min_z = bounds.low[2]
        max_x = bounds.high[0]
        max_y = bounds.high[1]
        max_z = bounds.high[2]
        p = self.rng.uniform([min_x, min_y, min_z], [max_x, max_y, max_z])
        subspace_state_out[0] = p[0]
        subspace_state_out[1] = p[1]
        subspace_state_out[2] = p[2]

    def sampleUniform(self, state_out: ob.CompoundState):
        # for i in range(2 + DualFloatingGripperRopeScenario.n_links):
        #     self.sample_point_for_R3_subspace(self.state_space.getSubspace(i), state_out[i])
        # state_np = self.scenario.ompl_state_to_numpy(state_out)

        random_point = self.rng.uniform(self.extent[:, 0], self.extent[:, 1])
        random_point_rope = np.concatenate([random_point] * DualFloatingGripperRopeScenario.n_links)
        state_np = {
            'gripper1': random_point,
            'gripper2': random_point,
            'link_bot': random_point_rope,
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }
        self.scenario.numpy_to_ompl_state(state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_state(state_np)


class DualGripperGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario: DualFloatingGripperRopeScenario,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(DualGripperGoalRegion, self).__init__(si)
        self.setThreshold(threshold)
        self.goal = goal
        self.scenario = scenario
        self.rng = rng
        self.plot = plot

    def distanceGoal(self, state: ob.CompoundState):
        """
        Uses the distance between a specific point in a specific subspace and the goal point
        """
        state_np = self.scenario.ompl_state_to_numpy(state)
        distance = self.scenario.distance_to_gripper_goal(state_np, self.goal)

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def sampleGoal(self, state_out: ob.CompoundState):
        sampler = self.getSpaceInformation().allocStateSampler()
        # sample a random state via the state space sampler, in hopes that OMPL will clean up the memory...
        sampler.sampleUniform(state_out)

        # don't bother trying to sample "legit" rope states, because this is only used to bias sampling towards the goal
        # so just prenteing every point on therope is at the goal should be sufficient
        rope = sample_rope_grippers(self.rng,
                                    self.goal['gripper1'],
                                    self.goal['gripper2'],
                                    DualFloatingGripperRopeScenario.n_links)

        goal_state_np = {
            'gripper1': self.goal['gripper1'],
            'gripper2': self.goal['gripper2'],
            'link_bot': rope.flatten(),
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }

        self.scenario.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100


class RopeMidpointGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario: DualFloatingGripperRopeScenario,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(RopeMidpointGoalRegion, self).__init__(si)
        self.setThreshold(threshold)
        self.goal = goal
        self.scenario = scenario
        self.rng = rng
        self.plot = plot

    def distanceGoal(self, state: ob.CompoundState):
        """
        Uses the distance between a specific point in a specific subspace and the goal point
        """
        state_np = self.scenario.ompl_state_to_numpy(state)
        distance = self.scenario.distance_to_midpoint_goal(state_np, self.goal)

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def sampleGoal(self, state_out: ob.CompoundState):
        sampler = self.getSpaceInformation().allocStateSampler()
        # sample a random state via the state space sampler, in hopes that OMPL will clean up the memory...
        sampler.sampleUniform(state_out)

        # attempt to sample "legit" rope states
        kd = 0.04
        rope = sample_rope(self.rng, self.goal['midpoint'], DualFloatingGripperRopeScenario.n_links, kd)
        # gripper 1 is attached to the last link
        gripper1 = rope[-1] + self.rng.uniform(-kd, kd, 3)
        gripper2 = rope[0] + self.rng.uniform(-kd, kd, 3)

        goal_state_np = {
            'gripper1': gripper1,
            'gripper2': gripper2,
            'link_bot': rope.flatten(),
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }

        self.scenario.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100


class RopeAnyPointGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario: DualFloatingGripperRopeScenario,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(RopeAnyPointGoalRegion, self).__init__(si)
        self.setThreshold(threshold)
        self.goal = goal
        self.scenario = scenario
        self.rng = rng
        self.plot = plot

    def distanceGoal(self, state: ob.CompoundState):
        """
        Uses the distance between a specific point in a specific subspace and the goal point
        """
        state_np = self.scenario.ompl_state_to_numpy(state)
        distance = self.scenario.distance_to_any_point_goal(state_np, self.goal)

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def sampleGoal(self, state_out: ob.CompoundState):
        sampler = self.getSpaceInformation().allocStateSampler()
        # sample a random state via the state space sampler, in hopes that OMPL will clean up the memory...
        sampler.sampleUniform(state_out)

        # attempt to sample "legit" rope states
        kd = 0.05
        rope = sample_rope(self.rng, self.goal['point'], DualFloatingGripperRopeScenario.n_links, kd)
        # gripper 1 is attached to the last link
        gripper1 = rope[-1] + self.rng.uniform(-kd, kd, 3)
        gripper2 = rope[0] + self.rng.uniform(-kd, kd, 3)

        goal_state_np = {
            'gripper1': gripper1,
            'gripper2': gripper2,
            'link_bot': rope.flatten(),
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }

        self.scenario.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100


class RopeAndGrippersGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario: DualFloatingGripperRopeScenario,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(RopeAndGrippersGoalRegion, self).__init__(si)
        self.setThreshold(threshold)
        self.goal = goal
        self.scenario = scenario
        self.rng = rng
        self.plot = plot

    def distanceGoal(self, state: ob.CompoundState):
        state_np = self.scenario.ompl_state_to_numpy(state)
        distance = self.scenario.distance_grippers_and_any_point_goal(state_np, self.goal)

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def sampleGoal(self, state_out: ob.CompoundState):
        # attempt to sample "legit" rope states
        kd = 0.05
        rope = sample_rope_and_grippers(
            self.rng, self.goal['gripper1'], self.goal['gripper2'], self.goal['point'],
            DualFloatingGripperRopeScenario.n_links,
            kd)

        goal_state_np = {
            'gripper1': self.goal['gripper1'],
            'gripper2': self.goal['gripper2'],
            'link_bot': rope.flatten(),
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }

        self.scenario.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100


class RopeAndGrippersBoxesGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario: DualFloatingGripperRopeScenario,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(RopeAndGrippersBoxesGoalRegion, self).__init__(si)
        self.goal = goal
        self.scenario = scenario
        self.setThreshold(threshold)
        self.rng = rng
        self.plot = plot

    def isSatisfied(self, state: ob.CompoundState, distance):
        state_np = self.scenario.ompl_state_to_numpy(state)
        rope_points = np.reshape(state_np['link_bot'], [-1, 3])
        n_from_ends = 7
        near_center_rope_points = rope_points[n_from_ends:-n_from_ends]

        gripper1_extent = np.reshape(self.goal['gripper1_box'], [3, 2])
        gripper1_satisfied = np.logical_and(
            state_np['gripper1'] >= gripper1_extent[:, 0], state_np['gripper1'] <= gripper1_extent[:, 1])

        gripper2_extent = np.reshape(self.goal['gripper2_box'], [3, 2])
        gripper2_satisfied = np.logical_and(
            state_np['gripper2'] >= gripper2_extent[:, 0], state_np['gripper2'] <= gripper2_extent[:, 1])

        point_extent = np.reshape(self.goal['point_box'], [3, 2])
        points_satisfied = np.logical_and(near_center_rope_points >=
                                          point_extent[:, 0], near_center_rope_points <= point_extent[:, 1])
        any_point_satisfied = np.reduce_any(points_satisfied)

        return float(any_point_satisfied and gripper1_satisfied and gripper2_satisfied)

    def sampleGoal(self, state_out: ob.CompoundState):
        # attempt to sample "legit" rope states
        kd = 0.05
        rope = sample_rope_and_grippers(
            self.rng, self.goal['gripper1'], self.goal['gripper2'], self.goal['point'],
            DualFloatingGripperRopeScenario.n_links,
            kd)

        goal_state_np = {
            'gripper1': self.goal['gripper1'],
            'gripper2': self.goal['gripper2'],
            'link_bot': rope.flatten(),
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }

        self.scenario.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_goal_state(goal_state_np)

    def distanceGoal(self, state: ob.CompoundState):
        state_np = self.scenario.ompl_state_to_numpy(state)
        distance = self.scenario.distance_grippers_and_any_point_goal(state_np, self.goal)

        # this ensures the goal must have num_diverged = 0
        if state_np['num_diverged'] > 0:
            distance = 1e9
        return distance

    def maxSampleCount(self):
        return 100
