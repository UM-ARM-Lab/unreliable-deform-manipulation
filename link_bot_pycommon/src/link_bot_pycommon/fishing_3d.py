from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from matplotlib import colors

import rospy
from geometry_msgs.msg import Pose, Point
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.link_bot_sdf_utils import environment_to_occupancy_msg
from link_bot_pycommon.params import CollectDynamicsParams
from link_bot_pycommon.pycommon import quaternion_from_euler
from moonshine.base_learned_dynamics_model import dynamics_loss_function, dynamics_points_metrics_function
from moonshine.moonshine_utils import remove_batch, add_batch
from mps_shape_completion_msgs.msg import OccupancyStamped
from peter_msgs.srv import Position3DAction, Position3DActionRequest, SetRopeConfiguration, Position3DEnableRequest, \
    SetRopeConfigurationRequest
from std_srvs.srv import Empty
from visualization_msgs.msg import MarkerArray, Marker


class Fishing3DScenario(ExperimentScenario):
    def __init__(self):
        super().__init__()
        object_name = 'link_bot'
        self.set_srv = rospy.ServiceProxy(f"{object_name}/set", Position3DAction)
        self.stop_object_srv = rospy.ServiceProxy(f"{object_name}/stop", Empty)
        self.object_enable_srv = rospy.ServiceProxy(f"{object_name}/enable", Empty)
        self.get_object_srv = rospy.ServiceProxy(f"{object_name}/get", Empty)
        self.set_rope_config_srv = rospy.ServiceProxy("set_rope_config", SetRopeConfiguration)

        self.env_viz_srv = rospy.Publisher('occupancy', OccupancyStamped, queue_size=10)
        self.state_viz_srv = rospy.Publisher("state_viz", MarkerArray, queue_size=10)
        self.action_viz_srv = rospy.Publisher("action_viz", MarkerArray, queue_size=10)

    def enable_object(self):
        enable_object = Position3DEnableRequest()
        enable_object.enable = True
        self.object_enable_srv(enable_object)

    def execute_action(self, action: Dict):
        req = Position3DActionRequest()
        req.position.x = action['position'][0]
        req.position.y = action['position'][1]
        req.position.z = action['position'][2]
        req.timeout = action['timeout'][0]
        _ = self.set_srv(req)

    def reset_rope(self, x, y, yaw, joint_angles):
        gripper_pose = Pose()
        gripper_pose.position.x = x
        gripper_pose.position.y = y
        q = quaternion_from_euler(0, 0, yaw)
        gripper_pose.orientation.x = q[0]
        gripper_pose.orientation.y = q[1]
        gripper_pose.orientation.z = q[2]
        gripper_pose.orientation.w = q[3]
        req = SetRopeConfigurationRequest()
        req.gripper_poses.append(gripper_pose)
        req.joint_angles.extend(joint_angles)
        self.set_rope_config_srv(req)

    def nudge(self):
        self.execute_action({
            'position': [np.random.randn(), np.random.randn(), 0],
            'timeout': [0.5],
        })

    @staticmethod
    def random_delta(state: Dict, action_rng: np.random.RandomState, environment, max_delta_pos=0.1):
        # sample a random point inside the bounds and generate an action in that direction of some max length
        target_pos = Fishing3DScenario.random_pos(action_rng, environment)
        current_pos = Fishing3DScenario.state_to_gripper_position(state)
        delta = target_pos - current_pos
        d = np.linalg.norm(delta)
        v = min(max_delta_pos, d)
        delta = delta / d * v
        return [delta[0], delta[1], delta[2]]

    @staticmethod
    def random_pos(action_rng: np.random.RandomState, environment):
        x_min, x_max, y_min, y_max, z_min, z_max = environment['extent']
        pos = action_rng.uniform([x_min, y_min, z_min], [x_max, y_max, z_max])
        return pos

    @staticmethod
    def sample_action(environment: Dict,
                      service_provider,
                      state,
                      last_action: Optional[Dict],
                      params: CollectDynamicsParams,
                      action_rng):
        # sample the previous action with 80% probability, this improves exploration
        if last_action is not None and action_rng.uniform(0, 1) < 0.80:
            return last_action
        else:
            return Fishing3DScenario.random_delta(state, action_rng, environment)

    def plot_environment_rviz(self, data: Dict):
        msg = environment_to_occupancy_msg(data)
        self.env_viz_srv.publish(msg)

    def plot_state_rviz(self, data: Dict, **kwargs):
        r, g, b, a = colors.to_rgba(kwargs.get("color", "r"))

        link_bot_points = np.reshape(data['link_bot'], [-1, 3])

        msg = MarkerArray()
        lines = Marker()
        lines.action = Marker.ADD  # create or modify
        lines.type = Marker.LINE_STRIP
        lines.header.frame_id = "/world"
        lines.header.stamp = rospy.Time.now()
        lines.ns = "link_bot"
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
        spheres.ns = "link_bot"
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
        self.state_viz_srv.publish(msg)

    def plot_action_rviz(self, data: Dict, **kwargs):
        # TODO:
        r, g, b, a = colors.to_rgba(kwargs.get("color", "r"))
        link_bot_points = np.reshape(data['link_bot'], [-1, 3])
        msg = MarkerArray()
        self.action_viz_srv.publish(msg)

    @staticmethod
    def state_to_gripper_position(state: Dict):
        gripper_position = np.reshape(state['gripper'], [3])
        return gripper_position

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
        return Fishing3DScenario.to_rope_local_frame_np(rope_state, reference_rope_state)

    @staticmethod
    def to_rope_local_frame_np(rope_state, reference_rope_state=None):
        if reference_rope_state is None:
            reference_rope_state = np.copy(rope_state)
        rope_local = Fishing3DScenario.to_rope_local_frame_tf(add_batch(rope_state), add_batch(reference_rope_state))
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

    def __repr__(self):
        return "Fishing"

    def simple_name(self):
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
    def integrate_dynamics(s_t, ds_t):
        return s_t + ds_t

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

    @staticmethod
    def put_state_local_frame(state_key, state):
        batch_size, time, _ = state.shape
        if state_key in ['link_bot', 'gripper']:
            points = tf.reshape(state, [batch_size, time, -1, 3])
            points = points - points[:, :, tf.newaxis, 0]
            state_in_local_frame = tf.reshape(points, [batch_size, time, -1])
            return state_in_local_frame
        else:
            raise NotImplementedError()
