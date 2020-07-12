from typing import Dict, Optional

import numpy as np
from time import sleep
import ros_numpy
import tensorflow as tf
import ompl.base as ob
import ompl.control as oc

import rospy
from link_bot_data.visualization import rviz_arrow
from geometry_msgs.msg import Point
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from matplotlib import colors
from link_bot_data.link_bot_dataset_utils import add_predicted
from tf import transformations
from moonshine.base_learned_dynamics_model import dynamics_loss_function, dynamics_points_metrics_function
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from visualization_msgs.msg import MarkerArray, Marker
from link_bot_pycommon.collision_checking import inflate_tf_3d
from link_bot_pycommon.pycommon import default_if_none, directions_3d
from victor_hardware_interface_msgs.msg import MotionCommand
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest, GetDualGripperPoints, WorldControlRequest, \
    SetRopeState, SetRopeStateRequest, SetDualGripperPoints, GetRopeState, GetRopeStateRequest, \
    GetDualGripperPointsRequest, SetBoolRequest, SetBool, GetJointState, GetJointStateRequest
from std_msgs.msg import Empty


# TODO: not floating
class DualFloatingGripperRopeScenario(Base3DScenario):
    n_links = 25

    def __init__(self):
        super().__init__()
        self.last_action = None
        self.action_srv = rospy.ServiceProxy("execute_dual_gripper_action", DualGripperTrajectory)
        self.grasping_rope_srv = rospy.ServiceProxy("set_grasping_rope", SetBool)
        self.get_grippers_srv = rospy.ServiceProxy("get_dual_gripper_points", GetDualGripperPoints)
        self.set_rope_srv = rospy.ServiceProxy("set_rope_state", SetRopeState)
        self.get_rope_srv = rospy.ServiceProxy("get_rope_state", GetRopeState)
        self.get_joint_state_srv = rospy.ServiceProxy("joint_states", GetJointState)
        self.set_grippers_srv = rospy.ServiceProxy("set_dual_gripper_points", SetDualGripperPoints)
        self.left_arm_motion_pub = rospy.Publisher("left_arm/motion_command", MotionCommand, queue_size=10)
        self.right_arm_motion_pub = rospy.Publisher("right_arm/motion_command", MotionCommand, queue_size=10)
        self.set_model_state_srv = rospy.ServiceProxy("gazebo/set_model_state", SetModelState)

        self.max_action_attempts = 500

        self.object_reset_poses = {
            'box1': (np.zeros(3), np.array([0, 0, 0, 1])),
            'box2': (np.zeros(3), np.array([0, 0, 0, 1])),
            'box3': (np.zeros(3), np.array([0, 0, 0, 1])),
            'box4': (np.zeros(3), np.array([0, 0, 0, 1])),
            'hook1': (np.zeros(3), np.array([0, 0, 0, 1])),
            'hook2': (np.zeros(3), np.array([0, 0, 0, 1])),
        }

    def reset_robot(self):
        rospy.logwarn("teleporting arms to home, ignoring obstacles!!!")
        left_arm_home = rospy.get_param("left_arm_home")
        right_arm_home = rospy.get_param("right_arm_home")

        left_arm_motion = MotionCommand()
        left_arm_motion.joint_position.joint_1 = left_arm_home[0]
        left_arm_motion.joint_position.joint_2 = left_arm_home[1]
        left_arm_motion.joint_position.joint_3 = left_arm_home[2]
        left_arm_motion.joint_position.joint_4 = left_arm_home[3]
        left_arm_motion.joint_position.joint_5 = left_arm_home[4]
        left_arm_motion.joint_position.joint_6 = left_arm_home[5]
        left_arm_motion.joint_position.joint_7 = left_arm_home[6]

        right_arm_motion = MotionCommand()
        right_arm_motion.joint_position.joint_1 = right_arm_home[0]
        right_arm_motion.joint_position.joint_2 = right_arm_home[1]
        right_arm_motion.joint_position.joint_3 = right_arm_home[2]
        right_arm_motion.joint_position.joint_4 = right_arm_home[3]
        right_arm_motion.joint_position.joint_5 = right_arm_home[4]
        right_arm_motion.joint_position.joint_6 = right_arm_home[5]
        right_arm_motion.joint_position.joint_7 = right_arm_home[6]

        for i in range(10):
            self.left_arm_motion_pub.publish(left_arm_motion)
            self.right_arm_motion_pub.publish(right_arm_motion)
            sleep(0.2)

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
                      action_params: Dict):
        action = None
        for _ in range(self.max_action_attempts):
            # move in the same direction as the previous action with some probability
            repeat_probability = data_collection_params['repeat_delta_gripper_motion_probability']
            if self.last_action is not None and action_rng.uniform(0, 1) < repeat_probability:
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
    def grippers_out_of_bounds(ripper1, gripper2, data_collection_params: Dict):
        gripper1_extent = data_collection_params['gripper1_action_sample_extent']
        gripper2_extent = data_collection_params['gripper2_action_sample_extent']
        return DualFloatingGripperRopeScenario.is_out_of_bounds(gripper1, gripper1_extent) \
            or DualFloatingGripperRopeScenario.is_out_of_bounds(gripper2, gripper2_extent)

    @ staticmethod
    def is_out_of_bounds(p, extent):
        x, y, z = p
        x_min, x_max, y_min, y_max, z_min, z_max = extent
        return x < x_min or x > x_max \
            or y < y_min or y > y_max \
            or z < z_min or z > z_max

    def settle(self):
        req = WorldControlRequest()
        req.seconds = 10
        self.world_control_srv(req)

    def randomize_environment(self, env_rng, objects_params: Dict, data_collection_params: Dict):
        # move the objects out of the way
        self.set_object_poses(self.object_reset_poses)

        # Let go of rope
        release = SetBoolRequest()
        release.data = False
        self.grasping_rope_srv(release)

        # teleport to home
        self.reset_robot()

        # re-grasp rope
        grasp = SetBoolRequest()
        grasp.data = True
        self.grasping_rope_srv(grasp)
        self.settle()

        # replace the objects in a new random configuration
        random_object_poses = {
            'box1': self.random_object_pose(env_rng, objects_params),
            'box2': self.random_object_pose(env_rng, objects_params),
            'box3': self.random_object_pose(env_rng, objects_params),
            'box4': self.random_object_pose(env_rng, objects_params),
            'hook1': self.random_object_pose(env_rng, objects_params),
            'hook2': self.random_object_pose(env_rng, objects_params),
        }
        self.set_object_poses(random_object_poses)

        extent = np.array(data_collection_params['extent']).reshape(3, 2)
        gripper1_position = env_rng.uniform(extent[:, 0], extent[:, 1])
        gripper2_position = env_rng.uniform(extent[:, 0], extent[:, 1])
        return_action = {
            'gripper1_position': gripper1_position,
            'gripper2_position': gripper2_position
        }
        self.execute_action(return_action)
        self.settle()

    def execute_action(self, action: Dict):
        target_gripper1_point = ros_numpy.msgify(Point, action['gripper1_position'])
        target_gripper2_point = ros_numpy.msgify(Point, action['gripper2_position'])

        req = DualGripperTrajectoryRequest()
        settling_time = rospy.get_param("world_interaction/traj_goal_time_tolerance")
        req.settling_time_seconds = settling_time
        req.gripper1_points.append(target_gripper1_point)
        req.gripper2_points.append(target_gripper2_point)
        res = self.action_srv(req)

    @ staticmethod
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

    @ staticmethod
    def local_environment_center_differentiable(state):
        rope_vector = state['link_bot']
        rope_points = tf.reshape(rope_vector, [rope_vector.shape[0], -1, 3])
        center = tf.reduce_mean(rope_points, axis=1)
        return center

    @ staticmethod
    def integrate_dynamics(s_t: Dict, delta_s_t: Dict):
        return {k: s_t[k] + delta_s_t[k] for k in s_t.keys()}

    @ staticmethod
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

    @ staticmethod
    def state_to_gripper_position(state: Dict):
        gripper_position1 = np.reshape(state['gripper1'], [3])
        gripper_position2 = np.reshape(state['gripper2'], [3])
        return gripper_position1, gripper_position2

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

        # joint_res = self.get_joint_state_srv(GetJointStateRequest())
        # victor_joint_names = joint_res.joint_state.name
        # victor_joint_positions = joint_res.joint_state.position

        return {
            'gripper1': ros_numpy.numpify(grippers_res.gripper1),
            'gripper2': ros_numpy.numpify(grippers_res.gripper2),
            'link_bot': np.array(rope_state_vector, np.float32),
            # 'rope_velocities': np.array(rope_velocity_vector, np.float32),
            # 'model_pose': model_pose,
            # 'joint_angles_axis1': np.array(rope_res.joint_angles_axis1, np.float32),
            # 'joint_angles_axis2': np.array(rope_res.joint_angles_axis2, np.float32),
            # 'victor_joint_names': victor_joint_names,
            # 'victor_joint_positions': victor_joint_positions,
        }

    @ staticmethod
    def states_description() -> Dict:
        return {
            'gripper1': 3,
            'gripper2': 3,
            'link_bot': DualFloatingGripperRopeScenario.n_links * 3,
        }

    @ staticmethod
    def actions_description() -> Dict:
        # should match the keys of the dict return from action_to_dataset_action
        return {
            'gripper1_position': 3,
            'gripper2_position': 3,
        }

    @staticmethod
    def state_to_points_for_cc(state: Dict):
        return state['link_bot'].reshape(-1, 3)

    @ staticmethod
    def index_predicted_state_time(state, t):
        state_t = {}
        for feature_name in ['gripper1', 'gripper2', 'link_bot']:
            state_t[feature_name] = state[add_predicted(feature_name)][:, t]
        return state_t

    @ staticmethod
    def index_state_time(state, t):
        state_t = {}
        for feature_name in ['gripper1', 'gripper2', 'link_bot']:
            state_t[feature_name] = state[feature_name][:, t]
        return state_t

    @ staticmethod
    def index_action_time(action, t):
        action_t = {}
        for feature_name in ['gripper1_position', 'gripper2_position']:
            if t < action[feature_name].shape[1]:
                action_t[feature_name] = action[feature_name][:, t]
            else:
                action_t[feature_name] = action[feature_name][:, t - 1]
        return action_t

    @ staticmethod
    def index_label_time(example: Dict, t: int):
        return example['is_close'][:, t]

    def __repr__(self):
        return "DualFloatingGripperRope"

    def simple_name(self):
        return "dual_floating_gripper_rope"

    @staticmethod
    def numpy_to_ompl_state(state_np: Dict, state_out: ob.CompoundState):
        for i in range(3):
            state_out[0][i] = np.float64(state_np['gripper1'][i])
        for i in range(3):
            state_out[1][i] = np.float64(state_np['gripper2'][i])
        for i in range(DualFloatingGripperRopeScenario.n_links * 3):
            state_out[2][i] = np.float64(state_np['link_bot'][i])
        state_out[3][0] = np.float64(state_np['stdev'][0])
        state_out[4][0] = np.float64(state_np['num_diverged'][0])

    @staticmethod
    def ompl_state_to_numpy(ompl_state: ob.CompoundState):
        gripper1 = np.array([ompl_state[0][0], ompl_state[0][1], ompl_state[0][2]])
        gripper2 = np.array([ompl_state[1][0], ompl_state[1][1], ompl_state[1][2]])
        rope = []
        for i in range(DualFloatingGripperRopeScenario.n_links):
            rope.append(ompl_state[2][3*i+0])
            rope.append(ompl_state[2][3*i+1])
            rope.append(ompl_state[2][3*i+2])
        rope = np.array(rope)
        return {
            'gripper1': gripper1,
            'gripper2': gripper2,
            'link_bot': rope,
            'stdev': np.array([ompl_state[3][0]]),
            'num_diverged': np.array([ompl_state[4][0]]),
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
    def sample_goal(environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        env_inflated = inflate_tf_3d(env=environment['env'],
                                     radius_m=planner_params['goal_threshold'], res=environment['res'])
        goal_extent = planner_params['goal_extent']

        while True:
            extent = np.array(goal_extent).reshape(3, 2)
            p = rng.uniform(extent[:, 0], extent[:, 1])
            goal = {'midpoint': p}
            row, col, channel = link_bot_sdf_utils.point_to_idx_3d_in_env(p[0], p[1], p[2], environment)
            collision = env_inflated[row, col, channel] > 0.5
            if not collision:
                return goal

    @staticmethod
    def distance_to_goal(state: Dict, goal: Dict):
        rope_points = np.reshape(state['link_bot'], [-1, 3])
        rope_midpoint = rope_points[int(DualFloatingGripperRopeScenario.n_links / 2)]
        distance = np.linalg.norm(goal['midpoint'] - rope_midpoint)
        return distance

    @staticmethod
    def distance_to_goal_differentiable(state, goal):
        rope_points = tf.reshape(state['link_bot'], [-1, 3])
        rope_midpoint = rope_points[int(DualFloatingGripperRopeScenario.n_links / 2)]
        distance = tf.linalg.norm(goal['midpoint'] - rope_midpoint)
        return distance

    @staticmethod
    def distance(s1: Dict, s2: Dict):
        """ this is not the distance metric used in planning, but the one used in evaluation (like distance to goal) """
        rope1_points = np.reshape(s1['link_bot'], [-1, 3])
        rope2_points = np.reshape(s2['link_bot'], [-1, 3])
        rope1_midpoint = rope1_points[int(DualFloatingGripperRopeScenario.n_links / 2)]
        rope2_midpoint = rope2_points[int(DualFloatingGripperRopeScenario.n_links / 2)]
        distance = float(np.linalg.norm(rope2_midpoint - rope1_midpoint))
        return distance

    @staticmethod
    def distance_differentiable(s1: Dict, s2: Dict):
        raise NotImplementedError()

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

    def make_goal_region(self, si: oc.SpaceInformation, rng: np.random.RandomState, params: Dict, goal: Dict, plot: bool):
        return DualGripperGoalRegion(si=si,
                                     scenario=self,
                                     rng=rng,
                                     threshold=params['goal_threshold'],
                                     goal=goal,
                                     plot=plot)

    def make_ompl_state_space(self, planner_params, state_sampler_rng: np.random.RandomState, plot: bool):
        state_space = ob.CompoundStateSpace()

        min_x, max_x, min_y, max_y, min_z, max_z = planner_params['extent']

        gripper1_subspace = ob.RealVectorStateSpace(3)
        gripper1_bounds = ob.RealVectorBounds(3)
        # these bounds are not used for sampling
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
        # these bounds are not used for sampling
        gripper2_bounds.setLow(0, min_x)
        gripper2_bounds.setHigh(0, max_x)
        gripper2_bounds.setLow(1, min_y)
        gripper2_bounds.setHigh(1, max_y)
        gripper2_bounds.setLow(2, min_z)
        gripper2_bounds.setHigh(2, max_z)
        gripper2_subspace.setBounds(gripper2_bounds)
        gripper2_subspace.setName("gripper2")
        state_space.addSubspace(gripper2_subspace, weight=1)

        rope_subspace = ob.RealVectorStateSpace(DualFloatingGripperRopeScenario.n_links * 3)
        rope_bounds = ob.RealVectorBounds(DualFloatingGripperRopeScenario.n_links * 3)
        # these bounds are not used for sampling
        rope_bounds.setLow(-1000)
        rope_bounds.setHigh(1000)
        rope_subspace.setBounds(rope_bounds)
        rope_subspace.setName("rope")
        state_space.addSubspace(rope_subspace, weight=1)

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
                                           extent=planner_params['extent'],
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

    def plot_goal(self, goal: Dict, goal_threshold: float, actually_at_goal: Optional[bool] = None):
        goal_marker_msg = MarkerArray()
        midpoint_marker = Marker()
        midpoint_marker.scale.x = goal_threshold * 2
        midpoint_marker.scale.y = goal_threshold * 2
        midpoint_marker.scale.z = goal_threshold * 2
        midpoint_marker.action = Marker.ADD
        midpoint_marker.type = Marker.SPHERE
        midpoint_marker.header.frame_id = "/world"
        midpoint_marker.header.stamp = rospy.Time.now()
        midpoint_marker.ns = 'goal'
        midpoint_marker.id = 0
        if actually_at_goal:
            midpoint_marker.color.r = 0.4
            midpoint_marker.color.g = 0.8
            midpoint_marker.color.b = 0.4
            midpoint_marker.color.a = 0.8
        else:
            midpoint_marker.color.r = 0.5
            midpoint_marker.color.g = 0.3
            midpoint_marker.color.b = 0.8
            midpoint_marker.color.a = 0.8
        midpoint_marker.pose.position.x = goal['midpoint'][0]
        midpoint_marker.pose.position.y = goal['midpoint'][1]
        midpoint_marker.pose.position.z = goal['midpoint'][2]
        midpoint_marker.pose.orientation.w = 1

        goal_marker_msg.markers.append(midpoint_marker)
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
        lines.header.frame_id = "/world"
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
        spheres.header.frame_id = "/world"
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
        gripper1_sphere.header.frame_id = "/world"
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
        gripper2_sphere.header.frame_id = "/world"
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
        gripper1_text.header.frame_id = "/world"
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
        midpoint_sphere.header.frame_id = "/world"
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

        idx1 = kwargs.get("idx1", 0)
        idx2 = kwargs.get("idx2", 1)

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


class DualGripperStateSampler(ob.RealVectorStateSampler):

    def __init__(self,
                 state_space,
                 scenario: DualFloatingGripperRopeScenario,
                 extent,
                 rng: np.random.RandomState,
                 plot: bool):
        super().__init__(state_space)
        self.scenario = scenario
        self.extent = np.array(extent).reshape(3, 2)
        self.rng = rng
        self.plot = plot

    def sampleUniform(self, state_out: ob.CompoundState):
        # trying to sample a "valid" rope state is difficult, and probably unimportant
        # because the only role this plays in planning is to cause exploration/expansion
        # by biasing towards regions of empty space. So here we just pick a random point
        # and duplicate it, as if all points on the rope were at this point
        random_point = self.rng.uniform(self.extent[:, 0], self.extent[:, 1])
        random_point_rope = np.concatenate([random_point]*DualFloatingGripperRopeScenario.n_links)
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
        distance = self.scenario.distance_to_goal(state_np, self.goal)

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
        rope = np.concatenate([self.goal['midpoint']]*DualFloatingGripperRopeScenario.n_links)

        goal_state_np = {
            'gripper1': self.goal['midpoint'],
            'gripper2': self.goal['midpoint'],
            'link_bot': rope,
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }

        self.scenario.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100
