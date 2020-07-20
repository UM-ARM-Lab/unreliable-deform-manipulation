from typing import Dict, Optional

import numpy as np
import ros_numpy
import tensorflow as tf
from matplotlib import colors
import ompl.base as ob
import ompl.control as oc

import rospy
from geometry_msgs.msg import Pose, Point
from link_bot_data.visualization import rviz_arrow
from link_bot_data.link_bot_dataset_utils import add_predicted
from link_bot_pycommon.collision_checking import inflate_tf_3d
from link_bot_data.visualization import plot_arrow, update_arrow
from link_bot_pycommon.ros_pycommon import make_movable_object_services
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.link_bot_sdf_utils import environment_to_occupancy_msg, point_to_idx_3d_in_env
from link_bot_pycommon.params import CollectDynamicsParams
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from moonshine.base_learned_dynamics_model import dynamics_loss_function, dynamics_points_metrics_function
from moonshine.moonshine_utils import remove_batch, add_batch
from mps_shape_completion_msgs.msg import OccupancyStamped
from peter_msgs.srv import GetRopeState, GetRopeStateRequest, Position3DAction, Position3DActionRequest, Position3DEnableRequest, Position3DEnable, GetPosition3D, GetPosition3DRequest, WorldControlRequest
from peter_msgs.srv import DualGripperTrajectory, DualGripperTrajectoryRequest, GetDualGripperPoints, GetDualGripperPointsRequest
from std_srvs.srv import EmptyRequest, Empty
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import MarkerArray, Marker


class RopeDraggingScenario(Base3DScenario):
    n_links = 10

    def __init__(self):
        super().__init__()
        object_name = 'link_bot'
        self.move_srv = rospy.ServiceProxy(f"{object_name}/move", Position3DAction)
        self.get_object_srv = rospy.ServiceProxy(f"{object_name}/get", GetPosition3D)

        self.action_srv = rospy.ServiceProxy("execute_dual_gripper_action", DualGripperTrajectory)
        self.get_grippers_srv = rospy.ServiceProxy("get_dual_gripper_points", GetDualGripperPoints)

        self.get_rope_srv = rospy.ServiceProxy("get_rope_state", GetRopeState)

        self.reset_sim_srv = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)

        self.last_action = None
        self.max_action_attempts = 1000

        self.movable_object_services = {}
        for i in range(1, 10):
            k = f'moving_box{i}'
            self.movable_object_services[k] = make_movable_object_services(k)

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
        lines.id = 3 * idx + 0

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
        spheres.action = Marker.ADD
        spheres.type = Marker.SPHERE_LIST
        spheres.header.frame_id = "/world"
        spheres.header.stamp = rospy.Time.now()
        spheres.ns = label
        spheres.id = 3 * idx + 1

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

        gripper_point = Point()
        gripper_point.x = state['gripper'][0]
        gripper_point.y = state['gripper'][1]
        gripper_point.z = state['gripper'][2]

        spheres.points.append(gripper_point)

        gripper = Marker()
        gripper.action = Marker.ADD
        gripper.type = Marker.SPHERE
        gripper.header.frame_id = "/world"
        gripper.header.stamp = rospy.Time.now()
        gripper.ns = label
        gripper.id = 3 * idx + 2

        gripper.scale.x = 0.02
        gripper.scale.y = 0.02
        gripper.scale.z = 0.02

        gripper.pose.position.x = state['gripper'][0]
        gripper.pose.position.y = state['gripper'][1]
        gripper.pose.position.z = state['gripper'][2]
        gripper.pose.orientation.x = 0
        gripper.pose.orientation.y = 0
        gripper.pose.orientation.z = 0
        gripper.pose.orientation.w = 1

        gripper.color.r = 1 - r
        gripper.color.g = 1 - g
        gripper.color.b = b
        gripper.color.a = a

        msg.markers.append(spheres)
        msg.markers.append(lines)
        msg.markers.append(gripper)
        self.state_viz_pub.publish(msg)

    def plot_action_rviz(self, state: Dict, action: Dict, label: str = 'action', **kwargs):
        state_action = {}
        state_action.update(state)
        state_action.update(action)
        self.plot_action_rviz_internal(state_action, label=label, **kwargs)

    def plot_action_rviz_internal(self, data: Dict, label: str, **kwargs):
        r, g, b, a = colors.to_rgba(kwargs.get("color", "b"))
        gripper = np.reshape(data['gripper'], [3])
        target_gripper = np.reshape(data['gripper_position'], [3])

        idx = kwargs.get("idx", 0)

        msg = MarkerArray()
        msg.markers.append(rviz_arrow(position=gripper,
                                      target_position=target_gripper,
                                      r=r, g=g, b=b, a=a,
                                      idx=idx, label=label, **kwargs))

        self.action_viz_pub.publish(msg)

    def val_execute_action(self, action: Dict):
        target_gripper1_point = ros_numpy.msgify(Point, action['gripper_position'])
        target_gripper1_point.z = max(target_gripper1_point.z, -0.38)
        grippers_res = self.get_grippers_srv(GetDualGripperPointsRequest())
        target_gripper2_point = grippers_res.gripper2

        req = DualGripperTrajectoryRequest()
        req.gripper1_points.append(target_gripper1_point)
        req.gripper2_points.append(target_gripper2_point)
        print(target_gripper1_point, target_gripper2_point)
        _ = self.action_srv(req)

    def execute_action(self, action: Dict):
        if rospy.get_param("use_val"):
            rospy.logwarn("TESTING WITH VAL")
            self.val_execute_action(action)
            return

        req = Position3DActionRequest()
        req.position.x = action['gripper_position'][0]
        req.position.y = action['gripper_position'][1]
        req.position.z = action['gripper_position'][2]
        req.timeout = action['timeout'][0]

        _ = self.move_srv(req)

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
        yaw = tf.random.uniform([batch_size, n_action_samples, n_actions], -np.pi, np.pi)
        max_d = action_params['max_distance_gripper_can_move']

        displacement = tf.random.uniform([batch_size, n_action_samples, n_actions], 0, max_d)

        zeros = tf.zeros([batch_size, n_action_samples, n_action_samples], dtype=tf.float32)

        gripper_delta_position = tf.stack([tf.math.sin(yaw), tf.math.cos(yaw), zeros], axis=3) * displacement

        # Apply delta
        gripper_position = state['gripper'][:, tf.newaxis, tf.newaxis] + gripper_delta_position

        actions = {
            'gripper_position': gripper_position,
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
            # sample the previous action with 80% probability, this improves exploration
            if self.last_action is not None and action_rng.uniform(0, 1) < 0.80:
                gripper_delta_position = self.last_action['gripper_delta_position']
            else:
                theta = action_rng.uniform(-np.pi, np.pi)
                displacement = action_rng.uniform(0, action_params['max_distance_gripper_can_move'])

                dx = np.cos(theta) * displacement
                dy = np.sin(theta) * displacement

                gripper_delta_position = np.array([dx, dy, 0])

            gripper_position = state['gripper'] + gripper_delta_position
            action = {
                'gripper_position': gripper_position,
                'gripper_delta_position': gripper_delta_position,
                'timeout': [action_params['dt']],
            }
            out_of_bounds = self.gripper_out_of_bounds(gripper_position, data_collection_params)
            if not out_of_bounds:
                self.last_action = action
                return action

        rospy.logwarn("Could not find a valid action, executing an invalid one")
        return action

    def gripper_out_of_bounds(self, gripper, data_collection_params: Dict):
        gripper_extent = data_collection_params['gripper_action_sample_extent']
        return RopeDraggingScenario.is_out_of_bounds(gripper, gripper_extent)

    @ staticmethod
    def is_out_of_bounds(p, extent):
        x, y, z = p
        x_min, x_max, y_min, y_max, z_min, z_max = extent
        return x < x_min or x > x_max \
            or y < y_min or y > y_max \
            or z < z_min or z > z_max

    def get_state_val(self):
        grippers_res = self.get_grippers_srv(GetDualGripperPointsRequest())

        rope_res = self.get_rope_srv(GetRopeStateRequest())

        rope_state_vector = []
        assert(len(rope_res.positions) == RopeDraggingScenario.n_links)
        for p in rope_res.positions:
            rope_state_vector.append(p.x)
            rope_state_vector.append(p.y)
            rope_state_vector.append(p.z)

        return {
            'gripper': ros_numpy.numpify(grippers_res.gripper1),
            'link_bot': np.array(rope_state_vector, np.float32),
        }

    def get_state(self):
        if rospy.get_param("use_val"):
            rospy.logwarn("TESTING WITH VAL")
            return self.get_state_val()

        gripper_res = self.get_object_srv(GetPosition3DRequest())

        rope_res = self.get_rope_srv(GetRopeStateRequest())

        rope_state_vector = []
        assert(len(rope_res.positions) == RopeDraggingScenario.n_links)
        for p in rope_res.positions:
            rope_state_vector.append(p.x)
            rope_state_vector.append(p.y)
            rope_state_vector.append(p.z)

        return {
            'gripper': ros_numpy.numpify(gripper_res.pos),
            'link_bot': np.array(rope_state_vector, np.float32),
        }

    @staticmethod
    def states_description() -> Dict:
        return {
            'gripper': 3,
            'link_bot': RopeDraggingScenario.n_links * 3,
        }

    @staticmethod
    def actions_description() -> Dict:
        # should match the keys of the dict return from action_to_dataset_action
        return {
            'gripper_position': 3,
            'timeout': 1,
        }

    @staticmethod
    def distance_to_goal(
            state: Dict[str, np.ndarray],
            goal: np.ndarray):
        """
        Uses the first point in the link_bot subspace as the thing which we want to move to goal
        :param state: A dictionary of numpy arrays
        :param goal: Assumed to be a point in 3D
        :return:
        """
        link_bot_points = np.reshape(state['link_bot'], [-1, 3])
        tail_point = link_bot_points[0]
        distance = np.linalg.norm(tail_point - goal['tail'])
        return distance

    @staticmethod
    def distance_to_goal_differentiable(state, goal):
        link_bot_points = tf.reshape(state['link_bot'], [-1, 3])[:, :2]
        tail_point = link_bot_points[0]
        distance = tf.linalg.norm(tail_point - goal)
        return distance

    @staticmethod
    def distance(s1, s2):
        link_bot_points1 = np.reshape(s1['link_bot'], [-1, 3])[:, :2]
        tail_point1 = link_bot_points1[0]
        link_bot_points2 = np.reshape(s2['link_bot'], [-1, 3])[:, :2]
        tail_point2 = link_bot_points2[0]
        return np.linalg.norm(tail_point1 - tail_point2)

    @staticmethod
    def distance_differentiable(s1, s2):
        link_bot_points1 = tf.reshape(s1['link_bot'], [-1, 3])
        tail_point1 = link_bot_points1[0]
        link_bot_points2 = tf.reshape(s2['link_bot'], [-1, 3])
        tail_point2 = link_bot_points2[0]
        return tf.linalg.norm(tail_point1 - tail_point2)

    @staticmethod
    def state_to_points_for_cc(state: Dict):
        return state['link_bot'].reshape(-1, 3)

    @staticmethod
    def sample_goal(environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        # We want goals to be very much in free space otherwise they're often not reachable
        env_inflated = inflate_tf_3d(env=environment['env'],
                                     radius_m=3 * planner_params['goal_threshold'], res=environment['res'])
        goal_extent = planner_params['goal_extent']

        while True:
            extent = np.array(goal_extent).reshape(3, 2)
            p = rng.uniform(extent[:, 0], extent[:, 1])
            goal = {'tail': p}
            row, col, channel = point_to_idx_3d_in_env(p[0], p[1], p[2], environment)
            collision = env_inflated[row, col, channel] > 0.5
            if not collision:
                return goal

    @staticmethod
    def local_environment_center_differentiable(state):
        return state['gripper']

    @staticmethod
    def __repr__():
        return "Rope Manipulation"

    @staticmethod
    def simple_name():
        return "link_bot"

    @staticmethod
    def robot_name():
        return "link_bot"

    @staticmethod
    def dynamics_loss_function(dataset_element, predictions):
        return dynamics_loss_function(dataset_element, predictions)

    @staticmethod
    def dynamics_metrics_function(dataset_element, predictions):
        return dynamics_points_metrics_function(dataset_element, predictions)

    @staticmethod
    def put_state_local_frame(state):
        rope = state['link_bot']
        rope_points_shape = rope.shape[:-1].as_list() + [-1, 3]
        points = tf.reshape(rope, rope_points_shape)
        center = state['gripper']
        rope_points_local = points - tf.expand_dims(center, axis=-2)
        gripper_local = state['gripper'] - center

        # # project all z coordinates down to what we saw in simulation... the real fix for this is to use TF and have actually cordinate frames
        # gripper_local = gripper_local * tf.constant([[1, 1, 0]], tf.float32) + tf.constant([[1, 1, 0.02]], tf.float32)
        # rope_points_local = rope_points_local * tf.constant([[1, 1, 0]], tf.float32)+ tf.constant([[1, 1, 0.02]], tf.float32)

        rope_local = tf.reshape(rope_points_local, rope.shape)

        return {
            'gripper': gripper_local,
            'link_bot': rope_local,
        }

    @ staticmethod
    def put_action_local_frame(state: Dict, action: Dict):
        target_gripper_position = action['gripper_position']

        current_gripper_point = state['gripper']

        gripper_delta = target_gripper_position - current_gripper_point

        return {
            'gripper_delta': gripper_delta,
        }

    def randomize_environment(self, env_rng, objects_params: Dict, data_collection_params: Dict):
        self.reset_sim_srv(EmptyRequest())

        # set random positions for all the objects
        for services in self.movable_object_services.values():
            position, _ = self.random_object_pose(env_rng, objects_params)
            set_msg = Position3DActionRequest()
            set_msg.position = ros_numpy.msgify(Point, position)
            services['set'](set_msg)

        req = WorldControlRequest()
        req.seconds = 0.2
        self.world_control_srv(req)

        for services in self.movable_object_services.values():
            disable = Position3DEnableRequest()
            disable.enable = False
            services['enable'](disable)

        req = WorldControlRequest()
        req.seconds = 0.2
        self.world_control_srv(req)

        for services in self.movable_object_services.values():
            services['stop'](EmptyRequest())

        req = WorldControlRequest()
        req.seconds = 0.2
        self.world_control_srv(req)

    @ staticmethod
    def integrate_dynamics(s_t: Dict, delta_s_t: Dict):
        return {k: s_t[k] + delta_s_t[k] for k in s_t.keys()}

    @ staticmethod
    def index_predicted_state_time(state, t):
        return {
            'gripper': state[add_predicted('gripper')][:, t],
            'link_bot': state[add_predicted('link_bot')][:, t],
        }

    @ staticmethod
    def index_state_time(state, t):
        return {
            'gripper': state['gripper'][:, t],
            'link_bot': state['link_bot'][:, t],
        }

    @ staticmethod
    def index_action_time(action, t):
        action_t = {}
        for feature_name in ['gripper_position']:
            if t < action[feature_name].shape[1]:
                action_t[feature_name] = action[feature_name][:, t]
            else:
                action_t[feature_name] = action[feature_name][:, t - 1]
        return action_t

    @ staticmethod
    def index_label_time(example: Dict, t: int):
        return example['is_close'][:, t]

    @staticmethod
    def compute_label(actual: Dict, predicted: Dict, labeling_params: Dict):
        actual_rope = np.array(actual["link_bot"])
        predicted_rope = np.array(predicted["link_bot"])
        model_error = np.linalg.norm(actual_rope - predicted_rope)
        threshold = labeling_params['threshold']
        is_close = model_error < threshold
        return is_close

    @staticmethod
    def numpy_to_ompl_state(state_np: Dict, state_out: ob.CompoundState):
        for i in range(3):
            state_out[0][i] = np.float64(state_np['gripper'][i])
        for i in range(RopeDraggingScenario.n_links * 3):
            state_out[1][i] = np.float64(state_np['link_bot'][i])
        state_out[2][0] = np.float64(state_np['stdev'][0])
        state_out[3][0] = np.float64(state_np['num_diverged'][0])

    @staticmethod
    def ompl_state_to_numpy(ompl_state: ob.CompoundState):
        gripper = np.array([ompl_state[0][0], ompl_state[0][1], ompl_state[0][2]])
        rope = []
        for i in range(RopeDraggingScenario.n_links):
            rope.append(ompl_state[1][3*i+0])
            rope.append(ompl_state[1][3*i+1])
            rope.append(ompl_state[1][3*i+2])
        rope = np.array(rope)
        return {
            'gripper': gripper,
            'link_bot': rope,
            'stdev': np.array([ompl_state[2][0]]),
            'num_diverged': np.array([ompl_state[3][0]]),
        }

    def ompl_control_to_numpy(self, ompl_state: ob.CompoundState, ompl_control: oc.CompoundControl):
        state_np = RopeDraggingScenario.ompl_state_to_numpy(ompl_state)
        current_gripper_position = state_np['gripper']

        gripper_delta_position = np.array([np.cos(ompl_control[0][0]) * ompl_control[0][1],
                                           np.sin(ompl_control[0][0])*ompl_control[0][1],
                                           0])
        target_gripper_position = current_gripper_position + gripper_delta_position
        return {
            'gripper_position': target_gripper_position,
            'timeout': [self.action_params['dt']],
        }

    def make_goal_region(self, si: oc.SpaceInformation, rng: np.random.RandomState, params: Dict, goal: Dict, plot: bool):
        return RopeDraggingGoalRegion(si=si,
                                      scenario=self,
                                      rng=rng,
                                      threshold=params['goal_threshold'],
                                      goal=goal,
                                      plot=plot)

    def make_ompl_state_space(self, planner_params, state_sampler_rng: np.random.RandomState, plot: bool):
        state_space = ob.CompoundStateSpace()

        min_x, max_x, min_y, max_y, min_z, max_z = planner_params['extent']

        gripper_subspace = ob.RealVectorStateSpace(3)
        gripper_bounds = ob.RealVectorBounds(3)
        # these bounds are not used for sampling
        gripper_bounds.setLow(0, min_x)
        gripper_bounds.setHigh(0, max_x)
        gripper_bounds.setLow(1, min_y)
        gripper_bounds.setHigh(1, max_y)
        gripper_bounds.setLow(2, min_z)
        gripper_bounds.setHigh(2, max_z)
        gripper_subspace.setBounds(gripper_bounds)
        gripper_subspace.setName("gripper")
        state_space.addSubspace(gripper_subspace, weight=1)

        rope_subspace = ob.RealVectorStateSpace(RopeDraggingScenario.n_links * 3)
        rope_bounds = ob.RealVectorBounds(RopeDraggingScenario.n_links * 3)
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
            return RopeDraggingStateSampler(state_space,
                                            scenario=self,
                                            extent=planner_params['extent'],
                                            rng=state_sampler_rng,
                                            plot=plot)

        state_space.setStateSamplerAllocator(ob.StateSamplerAllocator(_state_sampler_allocator))

        return state_space

    def make_ompl_control_space(self, state_space, rng: np.random.RandomState, action_params: Dict):
        control_space = oc.CompoundControlSpace(state_space)

        gripper_control_space = oc.RealVectorControlSpace(state_space, 3)
        gripper_control_bounds = ob.RealVectorBounds(3)
        # Direction (in XY plane)
        gripper_control_bounds.setLow(1, -np.pi)
        gripper_control_bounds.setHigh(1, np.pi)
        # Displacement
        self.action_params = action_params  # FIXME: terrible API
        max_d = action_params['max_distance_gripper_can_move']
        gripper_control_bounds.setLow(2, 0)
        gripper_control_bounds.setHigh(2, max_d)
        gripper_control_space.setBounds(gripper_control_bounds)
        control_space.addSubspace(gripper_control_space)

        def _allocator(cs):
            return RopeDraggingControlSampler(cs, scenario=self, rng=rng, action_params=action_params)

        # I override the sampler here so I can use numpy RNG to make things more deterministic.
        # ompl does not allow resetting of seeds, which causes problems when evaluating multiple
        # planning queries in a row.
        control_space.setControlSamplerAllocator(oc.ControlSamplerAllocator(_allocator))

        return control_space

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

    def plot_goal(self, goal: Dict, goal_threshold: float, actually_at_goal: Optional[bool] = None):
        goal_marker_msg = MarkerArray()
        tail_marker = Marker()
        tail_marker.scale.x = goal_threshold * 2
        tail_marker.scale.y = goal_threshold * 2
        tail_marker.scale.z = goal_threshold * 2
        tail_marker.action = Marker.ADD
        tail_marker.type = Marker.SPHERE
        tail_marker.header.frame_id = "/world"
        tail_marker.header.stamp = rospy.Time.now()
        tail_marker.ns = 'goal'
        tail_marker.id = 0
        if actually_at_goal:
            tail_marker.color.r = 0.4
            tail_marker.color.g = 0.8
            tail_marker.color.b = 0.4
            tail_marker.color.a = 0.8
        else:
            tail_marker.color.r = 0.5
            tail_marker.color.g = 0.3
            tail_marker.color.b = 0.8
            tail_marker.color.a = 0.8
        tail_marker.pose.position.x = goal['tail'][0]
        tail_marker.pose.position.y = goal['tail'][1]
        tail_marker.pose.position.z = goal['tail'][2]
        tail_marker.pose.orientation.w = 1

        goal_marker_msg.markers.append(tail_marker)
        self.state_viz_pub.publish(goal_marker_msg)


class RopeDraggingControlSampler(oc.ControlSampler):
    def __init__(self,
                 control_space: oc.CompoundControlSpace,
                 scenario: RopeDraggingScenario,
                 rng: np.random.RandomState,
                 action_params: Dict):
        super().__init__(control_space)
        self.scenario = scenario
        self.rng = rng
        self.control_space = control_space
        self.action_params = action_params

    def sampleNext(self, control_out, previous_control, state):
        # Direction
        yaw = self.rng.uniform(-np.pi, np.pi)
        # Displacement
        displacement = self.rng.uniform(0, self.action_params['max_distance_gripper_can_move'])

        control_out[0][0] = yaw
        control_out[0][1] = displacement

    def sampleStepCount(self, min_steps, max_steps):
        step_count = self.rng.randint(min_steps, max_steps)
        return step_count


class RopeDraggingStateSampler(ob.RealVectorStateSampler):

    def __init__(self,
                 state_space,
                 scenario: RopeDraggingScenario,
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
        random_point_rope = np.concatenate([random_point]*RopeDraggingScenario.n_links)
        state_np = {
            'gripper': random_point,
            'link_bot': random_point_rope,
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }

        self.scenario.numpy_to_ompl_state(state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_state(state_np)


class RopeDraggingGoalRegion(ob.GoalSampleableRegion):

    def __init__(self,
                 si: oc.SpaceInformation,
                 scenario: RopeDraggingScenario,
                 rng: np.random.RandomState,
                 threshold: float,
                 goal: Dict,
                 plot: bool):
        super(RopeDraggingGoalRegion, self).__init__(si)
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
        rope = np.concatenate([self.goal['tail']]*RopeDraggingScenario.n_links)

        goal_state_np = {
            'gripper': self.goal['tail'],
            'link_bot': rope,
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev': np.zeros(1, dtype=np.float64),
        }

        self.scenario.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100
