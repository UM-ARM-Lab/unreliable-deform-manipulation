from typing import Dict, Optional

import numpy as np
import ros_numpy
import tensorflow as tf
from matplotlib import colors

import rospy
from geometry_msgs.msg import Pose, Point
from link_bot_data.visualization import rviz_arrow
from link_bot_data.link_bot_dataset_utils import add_predicted
from link_bot_data.visualization import plot_arrow, update_arrow
from link_bot_pycommon.ros_pycommon import make_movable_object_services
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.link_bot_sdf_utils import environment_to_occupancy_msg
from link_bot_pycommon.params import CollectDynamicsParams
from link_bot_pycommon.base_3d_scenario import Base3DScenario
from moonshine.base_learned_dynamics_model import dynamics_loss_function, dynamics_points_metrics_function
from moonshine.moonshine_utils import remove_batch, add_batch
from mps_shape_completion_msgs.msg import OccupancyStamped
from peter_msgs.srv import GetRopeState, GetRopeStateRequest, Position3DAction, Position3DActionRequest, Position3DEnableRequest, Position3DEnable, GetPosition3D, GetPosition3DRequest, WorldControlRequest
from std_srvs.srv import EmptyRequest, Empty
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import MarkerArray, Marker


class RopeDraggingScenario(Base3DScenario):
    n_links = 10

    def __init__(self):
        super().__init__()
        object_name = 'link_bot'
        self.move_srv = rospy.ServiceProxy(f"{object_name}/move", Position3DAction)
        self.object_enable_srv = rospy.ServiceProxy(f"{object_name}/enable", Position3DEnable)
        self.get_object_srv = rospy.ServiceProxy(f"{object_name}/get", GetPosition3D)
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
        lines.id = 2 * idx + 0

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
        spheres.id = 2 * idx + 1

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

        msg.markers.append(spheres)
        msg.markers.append(lines)
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

    def execute_action(self, action: Dict):
        req = Position3DActionRequest()
        req.position.x = action['gripper_position'][0]
        req.position.y = action['gripper_position'][1]
        req.position.z = action['gripper_position'][2]
        req.timeout = action['timeout'][0]

        _ = self.move_srv(req)

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

    def get_state(self):
        gripper_res = self.get_object_srv(GetPosition3DRequest())
        rope_res = self.get_rope_srv(GetRopeStateRequest())

        rope_state_vector = []
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
        link_bot_points = np.reshape(state['link_bot'], [-1, 3])[:, :2]
        tail_point = link_bot_points[0]
        distance = np.linalg.norm(tail_point - goal)
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

    # @staticmethod
    # def sample_goal(state, goal):
    #     link_bot_state = state['link_bot']
    #     goal_points = np.reshape(link_bot_state, [-1, 3])
    #     goal_points -= goal_points[0]
    #     goal_points += goal
    #     goal_state = goal_points.flatten()
    #     return {
    #         'link_bot': goal_state
    #     }

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
        rope_local = tf.reshape(rope_points_local, rope.shape)
        gripper_local = state['gripper'] - center

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

    def randomize_environment(self, env_rng, objects_params: Dict):
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
        state_t = {}
        for feature_name in ['gripper', 'link_bot']:
            state_t[feature_name] = state[add_predicted(feature_name)][:, t]
        return state_t

    @ staticmethod
    def index_state_time(state, t):
        state_t = {}
        for feature_name in ['gripper', 'link_bot']:
            state_t[feature_name] = state[feature_name][:, t]
        return state_t

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

    # @staticmethod
    # def get_environment_from_state_dict(start_states: Dict):
    #     return {}

    # @staticmethod
    # def get_environment_from_example(example: Dict):
    #     if isinstance(example, tuple):
    #         example = example[0]

    #     return {
    #         'env': example['env'],
    #         'origin': example['origin'],
    #         'res': example['res'],
    #         'extent': example['extent'],
    #     }
