from typing import Dict

import numpy as np
import tensorflow as tf
from matplotlib import colors

import rospy
from geometry_msgs.msg import Point
from link_bot_data.visualization import rviz_arrow
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.link_bot_sdf_utils import environment_to_occupancy_msg
from moonshine.base_learned_dynamics_model import dynamics_loss_function, dynamics_points_metrics_function
from moonshine.moonshine_utils import remove_batch, add_batch
from mps_shape_completion_msgs.msg import OccupancyStamped
from visualization_msgs.msg import MarkerArray, Marker


class Base3DScenario(ExperimentScenario):
    def __init__(self, params: Dict):
        super().__init__(params)
        self.env_viz_pub = rospy.Publisher('occupancy', OccupancyStamped, queue_size=10)
        self.state_viz_pub = rospy.Publisher("state_viz", MarkerArray, queue_size=10)
        self.action_viz_pub = rospy.Publisher("action_viz", MarkerArray, queue_size=10)

    @staticmethod
    def random_pos(action_rng: np.random.RandomState, environment):
        x_min, x_max, y_min, y_max, z_min, z_max = environment['extent']
        pos = action_rng.uniform([x_min, y_min, z_min], [x_max, y_max, z_max])
        return pos

    def plot_environment_rviz(self, data: Dict):
        msg = environment_to_occupancy_msg(data)
        self.env_viz_pub.publish(msg)

    def plot_state_rviz(self, data: Dict, label: str, **kwargs):
        r, g, b, a = colors.to_rgba(kwargs.get("color", "r"))

        link_bot_points = np.reshape(data['link_bot'], [-1, 3])

        msg = MarkerArray()
        lines = Marker()
        lines.action = Marker.ADD  # create or modify
        lines.type = Marker.LINE_STRIP
        lines.header.frame_id = "/world"
        lines.header.stamp = rospy.Time.now()
        lines.ns = label
        lines.id = 0

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
        spheres.id = 1

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

        msg.markers.append(spheres)
        msg.markers.append(lines)
        self.state_viz_pub.publish(msg)

    def plot_action_rviz(self, data: Dict, **kwargs):
        r, g, b, a = colors.to_rgba(kwargs.get("color", "b"))
        s1 = np.reshape(data['gripper1'], [3])
        s2 = np.reshape(data['gripper2'], [3])
        a1 = np.reshape(data['gripper1_delta'], [3])
        a2 = np.reshape(data['gripper2_delta'], [3])

        msg = MarkerArray()
        msg.markers.append(rviz_arrow(s1, a1, r, g, b, a, idx=0))
        msg.markers.append(rviz_arrow(s2, a2, r, g, b, a, idx=1))

        self.action_viz_pub.publish(msg)

    @staticmethod
    def get_subspace_weight(subspace_name: str):
        if subspace_name == 'link_bot':
            return 1.0
        elif subspace_name == 'tether':
            return 0.0
        else:
            raise NotImplementedError("invalid subspace_name {}".format(subspace_name))

    @staticmethod
    def sample_goal(state, goal):
        link_bot_state = state['link_bot']
        goal_points = np.reshape(link_bot_state, [-1, 3])
        goal_points -= goal_points[0]
        goal_points += goal
        goal_state = goal_points.flatten()
        return {
            'link_bot': goal_state
        }

    @staticmethod
    def to_rope_local_frame(state, reference_state=None):
        rope_state = state['link_bot']
        if reference_state is None:
            reference_rope_state = np.copy(rope_state)
        else:
            reference_rope_state = reference_state['link_bot']
        return Base3DScenario.to_rope_local_frame_np(rope_state, reference_rope_state)

    @staticmethod
    def to_rope_local_frame_np(rope_state, reference_rope_state=None):
        if reference_rope_state is None:
            reference_rope_state = np.copy(rope_state)
        rope_local = Base3DScenario.to_rope_local_frame_tf(add_batch(rope_state), add_batch(reference_rope_state))
        return remove_batch(rope_local).numpy()

    @staticmethod
    def to_rope_local_frame_tf(rope_state, reference_rope_state=None):
        if reference_rope_state is None:
            # identity applies the identity transformation, i.e. copies
            reference_rope_state = tf.identity(rope_state)

        batch_size = rope_state.shape[0]
        rope_points = tf.reshape(rope_state, [batch_size, -1, 3])
        reference_rope_points = tf.reshape(reference_rope_state, [batch_size, -1, 3])

        # translate
        rope_points -= reference_rope_points[:, tf.newaxis, -1]

        rope_points = tf.reshape(rope_points, [batch_size, -1])
        return rope_points

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
    def get_environment_from_state_dict(start_states: Dict):
        return {}

    @staticmethod
    def get_environment_from_example(example: Dict):
        if isinstance(example, tuple):
            example = example[0]

        return {
            'env': example['env'],
            'origin': example['origin'],
            'res': example['res'],
            'extent': example['extent'],
        }