import warnings
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from matplotlib import colors

from link_bot_data.dataset_utils import add_predicted, get_maybe_predicted
from link_bot_gazebo_python.gazebo_services import gz_scope
from link_bot_gazebo_python.position_3d import Position3D
from link_bot_pycommon.base_services import BaseServices
from link_bot_pycommon.make_rope_markers import make_rope_marker, make_gripper_marker
from link_bot_pycommon.marker_index_generator import marker_index_generator
from rosgraph.names import ns_join

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    import ompl.base as ob
    import ompl.control as oc

import ros_numpy
import rospy
from gazebo_msgs.srv import SetModelStateRequest
from geometry_msgs.msg import Point
from link_bot_data.visualization import rviz_arrow
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from link_bot_pycommon.collision_checking import inflate_tf_3d
from link_bot_pycommon.grid_utils import point_to_idx_3d_in_env
from link_bot_pycommon.ros_pycommon import make_movable_object_services, get_environment_for_extents_3d
from moonshine.base_learned_dynamics_model import dynamics_loss_function, dynamics_points_metrics_function
from peter_msgs.srv import *
from std_srvs.srv import EmptyRequest, Empty
from visualization_msgs.msg import MarkerArray, Marker

rope_key_name = 'link_bot'
gazebo_model_name = "dragging_rope"


class RopeDraggingScenario(Base3DScenario):
    n_links = 10
    ROPE_NAMESPACE = 'dragging_rope'
    ROPE_LINK_NAME = gz_scope(ROPE_NAMESPACE, 'gripper1')

    def __init__(self):
        super().__init__()
        self.service_provider = BaseServices()

        self.pos3d = Position3D()
        self.get_rope_srv = rospy.ServiceProxy(ns_join(self.ROPE_NAMESPACE, "get_rope_state"), GetRopeState)
        self.reset_sim_srv = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)

        self.last_action = None
        self.max_action_attempts = 1000

        self.movable_object_services = {}
        for i in range(1, 10):
            k = f'moving_box{i}'
            self.movable_object_services[k] = make_movable_object_services(k)

    def __repr__(self):
        return "rope_dragging_scenario"

    def plot_state_rviz(self, state: Dict, label: str, **kwargs):
        r, g, b, a = colors.to_rgba(kwargs.get("color", "r"))
        idx = kwargs.get("idx", 0)
        ig = marker_index_generator(idx)

        msg = MarkerArray()
        if rope_key_name in state:
            rope_points = np.reshape(state[rope_key_name], [-1, 3])
            markers = make_rope_marker(rope_points, 'world', label + "_gt_" + rope_key_name, next(ig), r, g, b, a)
            msg.markers.extend(markers)

        if 'gripper' in state:
            gripper = state['gripper']
            gripper_sphere = make_gripper_marker(gripper, next(ig), r, g, b, a, label + 'gt_gripper', Marker.SPHERE)
            msg.markers.append(gripper_sphere)

        if add_predicted(rope_key_name) in state:
            rope_points = np.reshape(state[add_predicted(rope_key_name)], [-1, 3])
            markers = make_rope_marker(rope_points, 'world', label + "_" + rope_key_name, next(ig), r, g, b, a)
            msg.markers.extend(markers)

        if add_predicted('gripper') in state:
            pred_gripper = state[add_predicted('gripper')]
            gripper_sphere = make_gripper_marker(pred_gripper, next(ig), r, g, b, a, label + "_gripper", Marker.SPHERE)
            msg.markers.append(gripper_sphere)

        self.state_viz_pub.publish(msg)

    def plot_action_rviz(self, state: Dict, action: Dict, label: str = 'action', **kwargs):
        state_action = {}
        state_action.update(state)
        state_action.update(action)
        self.plot_action_rviz_internal(state_action, label=label, **kwargs)

    def plot_action_rviz_internal(self, data: Dict, label: str, **kwargs):
        r, g, b, a = colors.to_rgba(kwargs.get("color", "b"))
        gripper = np.reshape(get_maybe_predicted(data, 'gripper'), [3])
        target_gripper = np.reshape(get_maybe_predicted(data, 'gripper_position'), [3])

        idx = kwargs.get("idx", 0)

        msg = MarkerArray()
        msg.markers.append(rviz_arrow(position=gripper,
                                      target_position=target_gripper,
                                      r=r, g=g, b=b, a=a,
                                      idx=idx, label=label, **kwargs))

        self.action_viz_pub.publish(msg)

    def on_before_get_state_or_execute_action(self):
        self.pos3d.register(RegisterPosition3DControllerRequest(scoped_link_name=self.ROPE_LINK_NAME,
                                                                controller_type='pid',
                                                                kp_vel=10.0,
                                                                kp_pos=10.0,
                                                                max_force=10.0,
                                                                max_vel=0.1, ))

    def execute_action(self, action: Dict):
        timeout_s = action.get('timeout_s', 1.0)
        speed_mps = action.get('speed', 0.1)
        req = Position3DActionRequest(scoped_link_name=self.ROPE_LINK_NAME,
                                      position=ros_numpy.msgify(Point, action['gripper_position']),
                                      speed_mps=speed_mps,
                                      timeout_s=timeout_s,
                                      )
        self.pos3d.move(req)

    def sample_action(self,
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state: Dict,
                      action_params: Dict,
                      validate: bool,
                      stateless: Optional[bool] = False):
        action = None
        for _ in range(self.max_action_attempts):
            # sample the previous action with 80% probability, this improves exploration
            if self.last_action is not None and action_rng.uniform(0, 1) < 0.80 and not stateless:
                gripper_delta_position = self.last_action['gripper_delta_position']
            else:
                theta = action_rng.uniform(-np.pi, np.pi)
                displacement = action_rng.uniform(0, action_params['max_distance_gripper_can_move'])

                dx = np.cos(theta) * displacement
                dy = np.sin(theta) * displacement

                gripper_delta_position = np.array([dx, dy, 0])

            gripper_position = state['gripper'] + gripper_delta_position
            action = {
                'gripper_position':       gripper_position,
                'gripper_delta_position': gripper_delta_position,
                'timeout':                [action_params['dt']],
            }

            if not validate or self.is_action_valid(action, action_params):
                self.last_action = action
                return action

        rospy.logwarn("Could not find a valid action, executing an invalid one")
        return action

    def is_action_valid(self, action: Dict, action_params: Dict):
        out_of_bounds = self.gripper_out_of_bounds(action['gripper_position'], action_params)
        return not out_of_bounds

    @staticmethod
    def interpolate(start_state, end_state, step_size=0.05):
        gripper_start = np.array(start_state['gripper'])
        gripper_end = np.array(end_state['gripper'])

        gripper_delta = gripper_end - gripper_start

        steps = np.round(np.linalg.norm(gripper_delta) / step_size).astype(np.int64)

        interpolated_actions = []
        for t in np.linspace(step_size, 1, steps):
            gripper_t = gripper_start + gripper_delta * t
            action = {
                'gripper_position': gripper_t,
            }
            interpolated_actions.append(action)

        return interpolated_actions

    @staticmethod
    def add_noise(action: Dict, noise_rng: np.random.RandomState):
        gripper_noise = noise_rng.normal(scale=0.01, size=[3])
        return {
            'gripper_position': action['gripper_position'] + gripper_noise
        }

    def gripper_out_of_bounds(self, gripper, data_collection_params: Dict):
        gripper_extent = data_collection_params['gripper_action_sample_extent']
        return RopeDraggingScenario.is_out_of_bounds(gripper, gripper_extent)

    @staticmethod
    def is_out_of_bounds(p, extent):
        x, y, z = p
        x_min, x_max, y_min, y_max, z_min, z_max = extent
        return x < x_min or x > x_max \
               or y < y_min or y > y_max \
               or z < z_min or z > z_max

    def get_state(self):
        gripper_res = self.pos3d.get(GetPosition3DRequest(scoped_link_name=self.ROPE_LINK_NAME))

        rope_res = self.get_rope_srv(GetRopeStateRequest())

        rope_state_vector = []
        assert (len(rope_res.positions) == RopeDraggingScenario.n_links)
        for p in rope_res.positions:
            rope_state_vector.append(p.x)
            rope_state_vector.append(p.y)
            rope_state_vector.append(p.z)

        return {
            'gripper':     ros_numpy.numpify(gripper_res.pos),
            rope_key_name: np.array(rope_state_vector, np.float32),
        }

    @staticmethod
    def states_description() -> Dict:
        return {
            'gripper':     3,
            rope_key_name: RopeDraggingScenario.n_links * 3,
        }

    @staticmethod
    def actions_description() -> Dict:
        # should match the keys of the dict return from action_to_dataset_action
        return {
            'gripper_position': 3,
            'timeout':          1,
        }

    @staticmethod
    def distance_to_goal(
            state: Dict[str, np.ndarray],
            goal: np.ndarray):
        """
        Uses the first point in the rope subspace as the thing which we want to move to goal
        :param state: A dictionary of numpy arrays
        :param goal: Assumed to be a point in 3D
        :return:
        """
        rope_points = np.reshape(state[rope_key_name], [-1, 3])
        tail_point = rope_points[0]
        distance = np.linalg.norm(tail_point - goal['tail'])
        return distance

    @staticmethod
    def distance_to_goal_differentiable(state, goal):
        rope_points = tf.reshape(state[rope_key_name], [-1, 3])
        tail_point = rope_points[0]
        distance = tf.linalg.norm(tail_point - goal)
        return distance

    def classifier_distance(self, s1: Dict, s2: Dict):
        model_error = np.linalg.norm(s1[rope_key_name] - s2[rope_key_name], axis=-1)
        return model_error

    @staticmethod
    def distance_differentiable(s1, s2):
        rope_points1 = tf.reshape(s1[rope_key_name], [-1, 3])
        tail_point1 = rope_points1[0]
        rope_points2 = tf.reshape(s2[rope_key_name], [-1, 3])
        tail_point2 = rope_points2[0]
        return tf.linalg.norm(tail_point1 - tail_point2)

    @staticmethod
    def state_to_points_for_cc(state: Dict):
        return state[rope_key_name].reshape(-1, 3)

    def sample_goal(self, environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        # add more inflating to reduce the number of truly unacheivable gols
        env_inflated = inflate_tf_3d(env=environment['env'],
                                     radius_m=2 * planner_params['goal_threshold'],
                                     res=environment['res'])
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
    def simple_name():
        return "rope dragging"

    @staticmethod
    def dynamics_loss_function(dataset_element, predictions):
        return dynamics_loss_function(dataset_element, predictions)

    @staticmethod
    def dynamics_metrics_function(dataset_element, predictions):
        return dynamics_points_metrics_function(dataset_element, predictions)

    @staticmethod
    def put_state_robot_frame(state: Dict):
        return state

    @staticmethod
    def put_state_local_frame(state):
        rope = state[rope_key_name]
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
            'gripper':     gripper_local,
            rope_key_name: rope_local,
        }

    @staticmethod
    def put_action_local_frame(state: Dict, action: Dict):
        target_gripper_position = action['gripper_position']

        current_gripper_point = state['gripper']

        gripper_delta = target_gripper_position - current_gripper_point

        return {
            'gripper_delta': gripper_delta,
        }

    def randomization_initialization(self):
        pass

    def randomize_environment(self, env_rng, objects_params: Dict, data_collection_params: Dict):
        # TODO: make scenarios take in a environment method,
        #  or make environment randomization methods take in the scenario,
        #  or just make scenarios more composable and have a few different (static) combinations that are hard-coded
        return self.slide_obstacles(env_rng, objects_params, data_collection_params)
        # return self.lift_then_shuffle(env_rng, objects_params, data_collection_params)

    def slide_obstacles(self, env_rng, objects_params: Dict, data_collection_params: Dict):
        # If we reset the sim we'd get less interesting/diverse obstacle configurations
        # but without resetting we can't have repeatable trials because the rope can get in the way differently
        # depending on where it ended up from the previous trial
        # self.reset_sim_srv(EmptyRequest())

        # set random positions for all the objects
        for services in self.movable_object_services.values():
            position, _ = self.random_object_pose(env_rng, objects_params)
            set_msg = Position3DActionRequest(position=ros_numpy.msgify(Point, position))
            services['set'](set_msg)

        req = WorldControlRequest()
        req.seconds = 0.1
        self.world_control_srv(req)

        for services in self.movable_object_services.values():
            services['enable'](Position3DEnableRequest(enable=False))

        req = WorldControlRequest()
        req.seconds = 0.2
        self.world_control_srv(req)

        for services in self.movable_object_services.values():
            services['stop'](EmptyRequest())

        req = WorldControlRequest()
        req.seconds = 0.2
        self.world_control_srv(req)

    def lift_then_shuffle(self, env_rng, objects_params: Dict, data_collection_params: Dict):
        # reset so the rope is straightened out
        self.reset_sim_srv(EmptyRequest())

        # disable the rope dragging controller
        self.pos3d.enable(Position3DEnableRequest(enable=False))

        # lift the rope up out of the way with SetModelState
        move_rope = SetModelStateRequest()
        move_rope.model_state.model_name = gazebo_model_name
        move_rope.model_state.pose.position.x = 0
        move_rope.model_state.pose.position.y = 0
        move_rope.model_state.pose.position.z = 1
        self.set_model_state_srv(move_rope)

        # use SetModelState to move the objects to random configurations
        random_object_poses = self.random_new_object_poses(env_rng, objects_params)
        self.set_object_poses(random_object_poses)

        # let things settle, the rope will drop.
        self.settle()

        # re-enable the rope dragging controller
        self.pos3d.enable(Position3DEnableRequest(enable=True))

        # move to a random position, to improve diversity of "starting" configurations,
        # and to also reduce the number of trials where the rope starts on top of an obstacle
        random_position, _ = self.random_pose_in_extents(env_rng, data_collection_params['rope_start_extents'])
        random_position[2] = 0.02
        self.execute_action({
            'gripper_position': random_position,
            'timeout':          [30],
        })

    @staticmethod
    def integrate_dynamics(s_t: Dict, delta_s_t: Dict):
        return {k: s_t[k] + delta_s_t[k] for k in s_t.keys()}

    def index_action_time(self, action, t):
        action_t = {}
        for feature_name in ['gripper_position']:
            if action[feature_name].is_batched:
                if t < action[feature_name].shape[0]:
                    action_t[feature_name] = action[feature_name][t]
                else:
                    action_t[feature_name] = action[feature_name][t - 1]
            else:
                if t < action[feature_name].shape[1]:
                    action_t[feature_name] = action[feature_name][:, t]
                else:
                    action_t[feature_name] = action[feature_name][:, t - 1]
        return action_t

    @staticmethod
    def numpy_to_ompl_state(state_np: Dict, state_out: ob.CompoundState):
        for i in range(3):
            state_out[0][i] = np.float64(state_np['gripper'][i])
        for i in range(RopeDraggingScenario.n_links * 3):
            state_out[1][i] = np.float64(state_np[rope_key_name][i])
        state_out[2][0] = np.float64(state_np['stdev'][0])
        state_out[3][0] = np.float64(state_np['num_diverged'][0])

    @staticmethod
    def ompl_state_to_numpy(ompl_state: ob.CompoundState):
        gripper = np.array([ompl_state[0][0], ompl_state[0][1], ompl_state[0][2]])
        rope = []
        for i in range(RopeDraggingScenario.n_links):
            rope.append(ompl_state[1][3 * i + 0])
            rope.append(ompl_state[1][3 * i + 1])
            rope.append(ompl_state[1][3 * i + 2])
        rope = np.array(rope)
        return {
            'gripper':      gripper,
            rope_key_name:  rope,
            'stdev':        np.array([ompl_state[2][0]]),
            'num_diverged': np.array([ompl_state[3][0]]),
        }

    def ompl_control_to_numpy(self, ompl_state: ob.CompoundState, ompl_control: oc.CompoundControl):
        state_np = RopeDraggingScenario.ompl_state_to_numpy(ompl_state)
        current_gripper_position = state_np['gripper']

        gripper_delta_position = np.array([np.cos(ompl_control[0][0]) * ompl_control[0][1],
                                           np.sin(ompl_control[0][0]) * ompl_control[0][1],
                                           0])
        target_gripper_position = current_gripper_position + gripper_delta_position
        return {
            'gripper_position': target_gripper_position,
            'timeout':          [self.action_params['dt']],
        }

    def make_goal_region(self, si: oc.SpaceInformation, rng: np.random.RandomState, params: Dict, goal: Dict,
                         plot: bool):
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
        ig = marker_index_generator(self.tree_action_idx)
        idx1 = next(ig)
        idx2 = next(ig)
        self.plot_action_rviz(state, action, label='tree', color=[r, g, b, a], idx1=idx1, idx2=idx2, **kwargs)
        self.tree_action_idx += 1

    def plot_executed_action(self, state: Dict, action: Dict, **kwargs):
        self.plot_action_rviz(state, action, label='executed action', color="#3876EB", idx1=1, idx2=1, **kwargs)

    def plot_goal_rviz(self, goal: Dict, goal_threshold: float, actually_at_goal: Optional[bool] = None):
        goal_marker_msg = MarkerArray()
        tail_marker = Marker()
        tail_marker.scale.x = goal_threshold * 2
        tail_marker.scale.y = goal_threshold * 2
        tail_marker.scale.z = goal_threshold * 2
        tail_marker.action = Marker.ADD
        tail_marker.type = Marker.SPHERE
        tail_marker.header.frame_id = "world"
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

    def get_environment(self, params: Dict, **kwargs):
        res = params.get("res", 0.01)
        return get_environment_for_extents_3d(extent=params['extent'],
                                              res=res,
                                              service_provider=self.service_provider,
                                              excluded_models=self.get_excluded_models_for_env())

    def get_excluded_models_for_env(self):
        return ['dragging_rope']


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
        random_point_rope = np.concatenate([random_point] * RopeDraggingScenario.n_links)
        state_np = {
            'gripper':      random_point,
            rope_key_name:  random_point_rope,
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev':        np.zeros(1, dtype=np.float64),
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
        rope = np.concatenate([self.goal['tail']] * RopeDraggingScenario.n_links)

        goal_state_np = {
            'gripper':      self.goal['tail'],
            rope_key_name:  rope,
            'num_diverged': np.zeros(1, dtype=np.float64),
            'stdev':        np.zeros(1, dtype=np.float64),
        }

        self.scenario.numpy_to_ompl_state(goal_state_np, state_out)

        if self.plot:
            self.scenario.plot_sampled_goal_state(goal_state_np)

    def maxSampleCount(self):
        return 100
