from typing import Dict

import numpy as np
import ros_numpy
import tensorflow as tf
import ompl.base as ob
import ompl.control as oc

import rospy
from geometry_msgs.msg import Point
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from link_bot_data.link_bot_dataset_utils import add_predicted
from tf import transformations
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from victor_hardware_interface_msgs.msg import MotionCommand
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest, GetDualGripperPoints, WorldControlRequest, \
    WorldControl, SetRopeState, SetRopeStateRequest, SetDualGripperPoints, GetRopeState, GetRopeStateRequest, \
    GetDualGripperPointsRequest, SetBoolRequest, SetBool
from std_msgs.msg import Empty


class DualFloatingGripperRopeScenario(Base3DScenario):
    def __init__(self, params: Dict):
        super().__init__(params)
        self.last_state = None
        self.last_action = None
        self.can_repeat_last_action = False
        self.action_srv = rospy.ServiceProxy("execute_dual_gripper_action", DualGripperTrajectory)
        self.grasping_rope_srv = rospy.ServiceProxy("set_grasping_rope", SetBool)
        self.interrupt = rospy.Publisher("interrupt_trajectory", Empty, queue_size=10)
        self.get_grippers_srv = rospy.ServiceProxy("get_dual_gripper_points", GetDualGripperPoints)
        self.set_rope_srv = rospy.ServiceProxy("set_rope_state", SetRopeState)
        self.get_rope_srv = rospy.ServiceProxy("get_rope_state", GetRopeState)
        self.set_grippers_srv = rospy.ServiceProxy("set_dual_gripper_points", SetDualGripperPoints)
        self.world_control_srv = rospy.ServiceProxy("world_control", WorldControl)
        self.left_arm_motion_pub = rospy.Publisher("left_arm/motion_command", MotionCommand, queue_size=10)
        self.right_arm_motion_pub = rospy.Publisher("right_arm/motion_command", MotionCommand, queue_size=10)
        self.set_model_state_srv = rospy.ServiceProxy("gazebo/set_model_state", SetModelState)

        self.nudge_rng = np.random.RandomState(0)

        self.max_action_attempts = 1000

        self.object_reset_positions = {
            'box1': np.zeros(3),
            'box2': np.zeros(3),
            'box3': np.zeros(3),
            'box4': np.zeros(3),
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

        self.left_arm_motion_pub.publish(left_arm_motion)
        self.right_arm_motion_pub.publish(right_arm_motion)

    def sample_action(self,
                      environment: Dict,
                      state,
                      params: Dict,
                      action_rng):
        action = None

        gripper1_point = state['gripper1']
        # the last link connects to gripper 1 at the moment
        rope_state_vector = state['link_bot']
        link_point = np.array([rope_state_vector[-3], rope_state_vector[-2], rope_state_vector[-1]])
        distance_between_gripper1_and_link = np.linalg.norm(gripper1_point - link_point)

        for _ in range(self.max_action_attempts):
            # move in the same direction as the previous action with some probability
            repeat_probability = self.params['repeat_delta_gripper_motion_probability']
            if self.can_repeat_last_action and action_rng.uniform(0, 1) < repeat_probability:
                last_delta_gripper_1 = self.last_action['gripper1_position'] - self.last_state['gripper1']
                last_delta_gripper_2 = self.last_action['gripper2_position'] - self.last_state['gripper2']
                gripper1_position = state['gripper1'] + last_delta_gripper_1
                gripper2_position = state['gripper2'] + last_delta_gripper_2
            else:
                gripper1_position, gripper2_position = self.random_nearby_position_action(
                    state, action_rng, environment)

            out_of_bounds = self.grippers_out_of_bounds(gripper1_position, gripper2_position)
            action = {
                'gripper1_position': gripper1_position,
                'gripper2_position': gripper2_position,
            }
            if not out_of_bounds:
                self.last_state = state
                self.last_action = action
                return action

        rospy.logwarn("Could not find a valid action, executing an invalid one")
        return action

    def grippers_out_of_bounds(self, gripper1, gripper2):
        gripper1_extent = self.params['gripper1_action_sample_extent']
        gripper2_extent = self.params['gripper2_action_sample_extent']
        return DualFloatingGripperRopeScenario.is_out_of_bounds(gripper1, gripper1_extent) \
            or DualFloatingGripperRopeScenario.is_out_of_bounds(gripper2, gripper2_extent)

    @ staticmethod
    def is_out_of_bounds(p, extent):
        x, y, z = p
        x_min, x_max, y_min, y_max, z_min, z_max = extent
        return x < x_min or x > x_max \
            or y < y_min or y > y_max \
            or z < z_min or z > z_max

    def random_delta_position(self, Dict, action_rng: np.random.RandomState, environment: Dict):
        max_d = self.params['max_distance_gripper_can_move']
        delta1 = action_rng.uniform([-max_d, -max_d, -max_d], [max_d, max_d, max_d])
        delta2 = action_rng.uniform([-max_d, -max_d, -max_d], [max_d, max_d, max_d])

        delta1 = delta1 / np.linalg.norm(delta1)
        gripper1_displacement = gripper1_displacement * max_d * action_rng.uniform(0, 1)
        target_gripper1_pos = current_gripper1_pos + gripper1_displacement

        gripper2_displacement = gripper2_displacement / np.linalg.norm(gripper2_displacement)
        gripper2_displacement = gripper2_displacement * max_d * action_rng.uniform(0, 1)
        target_gripper2_pos = current_gripper2_pos + gripper2_displacement

        return target_gripper1_pos, target_gripper2_pos

    def random_nearby_position_action(self, state: Dict, action_rng: np.random.RandomState, environment: Dict):
        max_d = self.params['max_distance_gripper_can_move']
        target_gripper1_pos = Base3DScenario.random_pos(action_rng, self.params['gripper1_action_sample_extent'])
        target_gripper2_pos = Base3DScenario.random_pos(action_rng, self.params['gripper2_action_sample_extent'])
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
        settling_time = rospy.get_param("world_interaction/traj_goal_time_tolerance")
        req.seconds = settling_time
        self.world_control_srv(req)

    def randomize_environment(self, env_rng):
        state = self.get_state()
        pre_randomize_gripper1_position = state['gripper1']
        pre_randomize_gripper2_position = state['gripper2']

        # move the objects out of the way
        self.set_object_positions(self.object_reset_positions)

        # Let go of rope
        release = SetBoolRequest()
        release.data = False
        self.grasping_rope_srv(release)

        # teleport to home
        self.reset_robot()

        # replace the objects in a new random configuration
        random_object_positions = {
            'box1': self.random_object_position(env_rng),
            'box2': self.random_object_position(env_rng),
            'box3': self.random_object_position(env_rng),
            'box4': self.random_object_position(env_rng),
        }
        self.set_object_positions(random_object_positions)

        # re-grasp rope
        grasp = SetBoolRequest()
        grasp.data = True
        self.grasping_rope_srv(grasp)
        self.settle()

        # try to move back, but add some noise so we don't get immediately stuck again
        noise1 = env_rng.randn(3) * 0.1
        noise2 = env_rng.randn(3) * 0.1
        return_action = {
            'gripper1_position': pre_randomize_gripper1_position + noise1,
            'gripper2_position': pre_randomize_gripper2_position + noise2,
        }
        self.execute_action(return_action)

    def random_object_position(self, env_rng: np.random.RandomState):
        extent = self.params['objects_extent']
        extent = np.array(extent).reshape(3, 2)
        return env_rng.uniform(extent[:, 0], extent[:, 1])

    def set_object_positions(self, object_positions: Dict):
        for object_name, position in object_positions.items():
            set_req = SetModelStateRequest()
            set_req.model_state.model_name = object_name
            set_req.model_state.pose.position.x = position[0]
            set_req.model_state.pose.position.y = position[1]
            set_req.model_state.pose.position.z = position[2]
            self.set_model_state_srv(set_req)

    def execute_action(self, action: Dict):
        target_gripper1_point = ros_numpy.msgify(Point, action['gripper1_position'])
        target_gripper2_point = ros_numpy.msgify(Point, action['gripper2_position'])

        req = DualGripperTrajectoryRequest()
        settling_time = rospy.get_param("world_interaction/traj_goal_time_tolerance")
        req.settling_time_seconds = settling_time
        req.gripper1_points.append(target_gripper1_point)
        req.gripper2_points.append(target_gripper2_point)
        res = self.action_srv(req)

    def nudge(self, state: Dict, environment: Dict):
        nudge_action = self.random_nearby_position_action(state, self.nudge_rng, environment)
        self.execute_action(nudge_action)

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
        """
        :param state: Dict of batched states
        :return:
        """
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

    @ staticmethod
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
        for i in range(45):
            state_out[2][i] = np.float64(state_np['link_bot'][i])
        state_out[3][0] = np.float64(state_np['stdev'][0])
        state_out[4][0] = np.float64(state_np['num_diverged'][0])

    @staticmethod
    def ompl_state_to_numpy(ompl_state: ob.CompoundState):
        gripper1 = np.array([ompl_state[0][0], ompl_state[0][1], ompl_state[0][2]])
        gripper2 = np.array([ompl_state[1][0], ompl_state[1][1], ompl_state[1][2]])
        rope = []
        for i in range(15):
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
    def sample_goal(extent, rng: np.random.RandomState):
        extent = np.array(extent).reshape(3, 2)
        return {
            'midpoint': rng.uniform(extent[:, 0], extent[:, 1]),
        }

    @staticmethod
    def distance_to_goal(state: Dict, goal: Dict):
        rope_points = np.reshape(state['link_bot'], [-1, 3])
        rope_midpoint = rope_points[7]
        distance = np.linalg.norm(goal['midpoint'] - rope_midpoint)
        return distance

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

        rope_subspace = ob.RealVectorStateSpace(45)
        rope_bounds = ob.RealVectorBounds(45)
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

    def make_ompl_control_space(self, state_space, params, rng: np.random.RandomState):
        control_space = oc.CompoundControlSpace(state_space)

        gripper1_control_space = oc.RealVectorControlSpace(state_space, 3)
        gripper1_control_bounds = ob.RealVectorBounds(3)
        # Roll
        gripper1_control_bounds.setLow(0, -np.pi)
        gripper1_control_bounds.setHigh(0, np.pi)
        # Pitch
        gripper1_control_bounds.setLow(1, -np.pi)
        gripper1_control_bounds.setHigh(1, np.pi)
        # Displacement
        max_d = self.params['max_distance_gripper_can_move']
        gripper1_control_bounds.setLow(2, 0)
        gripper1_control_bounds.setHigh(2, max_d)
        gripper1_control_space.setBounds(gripper1_control_bounds)
        control_space.addSubspace(gripper1_control_space)

        gripper2_control_space = oc.RealVectorControlSpace(state_space, 3)
        gripper2_control_bounds = ob.RealVectorBounds(3)
        # Roll
        gripper2_control_bounds.setLow(0, -np.pi)
        gripper2_control_bounds.setHigh(0, np.pi)
        # Pitch
        gripper2_control_bounds.setLow(1, -np.pi)
        gripper2_control_bounds.setHigh(1, np.pi)
        # Displacement
        max_d = self.params['max_distance_gripper_can_move']
        gripper2_control_bounds.setLow(2, 0)
        gripper2_control_bounds.setHigh(2, max_d)

        gripper2_control_space.setBounds(gripper2_control_bounds)
        control_space.addSubspace(gripper2_control_space)

        return control_space


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
        random_point_rope = np.concatenate([random_point]*15)
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
        self.rope_link_length = 0.04
        self.rng = rng
        self.plot = plot

    def sampleGoal(self, state_out: ob.CompoundState):
        sampler = self.getSpaceInformation().allocStateSampler()
        # sample a random state via the state space sampler, in hopes that OMPL will clean up the memory...
        sampler.sampleUniform(state_out)

        # don't bother trying to sample "legit" rope states, because this is only used to bias sampling towards the goal
        # so just prenteing every point on therope is at the goal should be sufficient
        rope = np.concatenate([self.goal['midpoint']]*15)

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
