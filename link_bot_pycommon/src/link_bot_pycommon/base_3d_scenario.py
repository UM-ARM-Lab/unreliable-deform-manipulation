from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from matplotlib import cm
import ompl.base as ob
from matplotlib import colors

import rospy
from geometry_msgs.msg import Point
from jsk_recognition_msgs.msg import BoundingBox
from link_bot_data.link_bot_dataset_utils import NULL_PAD_VALUE
from link_bot_data.visualization import rviz_arrow
from tf import transformations
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.link_bot_sdf_utils import environment_to_occupancy_msg, extent_to_env_size
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from peter_msgs.msg import LabelStatus
from peter_msgs.srv import WorldControl
from moonshine.moonshine_utils import remove_batch, add_batch
from mps_shape_completion_msgs.msg import OccupancyStamped
from std_msgs.msg import Float32, Int64
from visualization_msgs.msg import MarkerArray, Marker


class Base3DScenario(ExperimentScenario):
    def __init__(self):
        super().__init__()
        self.world_control_srv = rospy.ServiceProxy("world_control", WorldControl)
        self.env_viz_pub = rospy.Publisher('occupancy', OccupancyStamped, queue_size=10, latch=True)
        self.env_bbox_pub = rospy.Publisher('env_bbox', BoundingBox, queue_size=10, latch=True)
        self.state_viz_pub = rospy.Publisher("state_viz", MarkerArray, queue_size=10, latch=True)
        self.action_viz_pub = rospy.Publisher("action_viz", MarkerArray, queue_size=10, latch=True)
        self.label_viz_pub = rospy.Publisher("label_viz", LabelStatus, queue_size=10, latch=True)
        self.traj_idx_viz_pub = rospy.Publisher("traj_idx_viz", Float32, queue_size=10, latch=True)
        self.time_viz_pub = rospy.Publisher("rviz_anim/time", Int64, queue_size=10, latch=True)
        self.accept_probability_viz_pub = rospy.Publisher("accept_probability_viz", Float32, queue_size=10, latch=True)
        try:
            import tf2_ros
            self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        except ImportError:
            self.broadcaster = None

        self.sampled_goal_marker_idx = 0
        self.tree_state_idx = 0
        self.rejected_state_idx = 0
        self.maybe_rejected_state_idx = 0
        self.current_tree_state_idx = 0
        self.tree_action_idx = 0
        self.sample_idx = 0

    @staticmethod
    def random_pos(action_rng: np.random.RandomState, extent):
        x_min, x_max, y_min, y_max, z_min, z_max = extent
        pos = action_rng.uniform([x_min, y_min, z_min], [x_max, y_max, z_max])
        return pos

    def reset_planning_viz(self):
        clear_msg = MarkerArray()
        clear_marker_msg = Marker()
        clear_marker_msg.action = Marker.DELETEALL
        clear_msg.markers.append(clear_marker_msg)
        from time import sleep
        for i in range(10):
            self.state_viz_pub.publish(clear_msg)
            self.action_viz_pub.publish(clear_msg)
            sleep(0.1)
        self.sampled_goal_marker_idx = 0
        self.tree_state_idx = 0
        self.rejected_state_idx = 0
        self.maybe_rejected_state_idx = 0
        self.current_tree_state_idx = 0
        self.tree_action_idx = 0
        self.sample_idx = 0

    def plot_environment_rviz(self, data: Dict):
        self.send_occupancy_tf(data)

        env_msg = environment_to_occupancy_msg(data)
        self.env_viz_pub.publish(env_msg)

        depth, width, height = extent_to_env_size(data['extent'])
        bbox_msg = BoundingBox()
        bbox_msg.header.frame_id = 'occupancy'
        bbox_msg.pose.position.x = width / 2
        bbox_msg.pose.position.y = depth / 2
        bbox_msg.pose.position.z = height / 2
        bbox_msg.dimensions.x = width
        bbox_msg.dimensions.y = depth
        bbox_msg.dimensions.z = height
        self.env_bbox_pub.publish(bbox_msg)

    def send_occupancy_tf(self, environment: Dict):
        link_bot_sdf_utils.send_occupancy_tf(self.broadcaster, environment)

    def plot_sampled_goal_state(self, state: Dict):
        self.plot_state_rviz(state, idx=self.sampled_goal_marker_idx, label="goal sample", color='#EB322F')
        self.sampled_goal_marker_idx += 1

    def plot_start_state(self, state: Dict):
        self.plot_state_rviz(state, label='start', color='#0088aa')

    def plot_sampled_state(self, state: Dict):
        self.plot_state_rviz(state, idx=self.sample_idx, label='samples', color='#f52f32')
        self.sample_idx += 1

    def plot_rejected_state(self, state: Dict):
        self.plot_state_rviz(state, idx=self.rejected_state_idx, label='rejected', color='#ff8822')
        self.rejected_state_idx += 1

    def plot_maybe_rejected_state(self, state: Dict):
        self.plot_state_rviz(state, idx=self.maybe_rejected_state_idx, label='rejected', color='#fac57f')
        self.maybe_rejected_state_idx += 1

    def plot_current_tree_state(self, state: Dict, horizon: int):
        if horizon is None:
            c = "#777777"
        else:
            c = cm.Oranges(state['num_diverged'][0] / horizon)
        self.plot_state_rviz(state, idx=1, label='current tree state', color=c)

    def plot_tree_state(self, state: Dict):
        self.plot_state_rviz(state, idx=self.tree_state_idx, label='tree', color='#777777')
        self.tree_state_idx += 1

    def plot_is_close(self, label_t):
        msg = LabelStatus()
        if label_t is None:
            msg.status = LabelStatus.NA
        elif label_t:
            msg.status = LabelStatus.Accept
        else:
            msg.status = LabelStatus.Reject
        self.label_viz_pub.publish(msg)

    def plot_accept_probability(self, accept_probability_t):
        msg = Float32()
        msg.data = accept_probability_t
        self.accept_probability_viz_pub.publish(msg)

    def plot_traj_idx_rviz(self, traj_idx):
        msg = Float32()
        msg.data = traj_idx
        self.traj_idx_viz_pub.publish(msg)

    def plot_time_idx_rviz(self, time_idx):
        msg = Int64()
        msg.data = time_idx
        self.time_viz_pub.publish(msg)

    def animate_evaluation_results(self,
                                   environment: Dict,
                                   actual_states: List[Dict],
                                   predicted_states: List[Dict],
                                   actions: List[Dict],
                                   goal: Dict,
                                   goal_threshold: float,
                                   labeling_params: Dict,
                                   accept_probabilities,
                                   horizon: int):
        time_steps = np.arange(len(actual_states))
        self.plot_environment_rviz(environment)
        self.plot_goal(goal, goal_threshold)

        anim = RvizAnimationController(time_steps)

        while not anim.done:
            t = anim.t()
            s_t = actual_states[t]
            s_t_pred = predicted_states[t]
            self.plot_state_rviz(s_t, label='actual', color='#ff0000aa')
            if horizon is None or 'num_diverged' not in s_t_pred:
                c = '#0000ffaa'
            else:
                c = cm.Blues(s_t_pred['num_diverged'][0] / horizon)
            self.plot_state_rviz(s_t_pred, label='predicted', color=c)
            if len(actions) > 0:
                if t < anim.max_t:
                    self.plot_action_rviz(s_t, actions[t])
                else:
                    self.plot_action_rviz(actual_states[t - 1], actions[t - 1])

            is_close = self.compute_label(s_t, s_t_pred, labeling_params)
            self.plot_is_close(is_close)

            actually_at_goal = self.distance_to_goal(s_t, goal) < goal_threshold
            self.plot_goal(goal, goal_threshold, actually_at_goal)

            if accept_probabilities and t > 0:
                self.plot_accept_probability(accept_probabilities[t - 1])
            else:
                self.plot_accept_probability(NULL_PAD_VALUE)

            anim.step()

    def animate_rviz(self,
                     environment: Dict,
                     actual_states: List[Dict],
                     predicted_states: List[Dict],
                     actions: List[Dict],
                     labels,
                     accept_probabilities):
        time_steps = np.arange(len(actual_states))
        self.plot_environment_rviz(environment)

        anim = RvizAnimationController(time_steps)

        while not anim.done:
            t = anim.t()
            s_t = actual_states[t]
            s_t_pred = predicted_states[t]
            self.plot_state_rviz(s_t, label='actual', color='#ff0000aa')
            self.plot_state_rviz(s_t_pred, label='predicted', color='#0000ffaa')
            if t < anim.max_t:
                self.plot_action_rviz(s_t, actions[t])
            else:
                self.plot_action_rviz(actual_states[t - 1], actions[t - 1])

            if labels is not None:
                self.plot_is_close(labels[t])

            if accept_probabilities and t > 0:
                self.plot_accept_probability(accept_probabilities[t - 1])
            else:
                self.plot_accept_probability(NULL_PAD_VALUE)

            anim.step()

    def animate_final_path(self,
                           environment: Dict,
                           planned_path: List[Dict],
                           actions: List[Dict]):
        time_steps = np.arange(len(planned_path))
        self.plot_environment_rviz(environment)

        anim = RvizAnimationController(time_steps)

        while not anim.done:
            t = anim.t()
            s_t_planned = planned_path[t]
            self.plot_state_rviz(s_t_planned, label='planned', color='#FF4616')
            if len(actions) > 0:
                if t < anim.max_t:
                    self.plot_action_rviz(s_t_planned, actions[t])
                else:
                    self.plot_action_rviz(planned_path[t - 1], actions[t - 1])

            anim.step()

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
    def robot_name():
        return "victor_and_rope::link_bot"

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

    def random_object_pose(self, env_rng: np.random.RandomState, objects_params: Dict):
        extent = objects_params['objects_extent']
        extent = np.array(extent).reshape(3, 2)
        position = env_rng.uniform(extent[:, 0], extent[:, 1])
        yaw = env_rng.uniform(-np.pi, np.pi)
        orientation = transformations.quaternion_from_euler(0, 0, yaw)
        return (position, orientation)

    def set_object_poses(self, object_positions: Dict):
        for object_name, (position, orientation) in object_positions.items():
            set_req = SetModelStateRequest()
            set_req.model_state.model_name = object_name
            set_req.model_state.pose.position.x = position[0]
            set_req.model_state.pose.position.y = position[1]
            set_req.model_state.pose.position.z = position[2]
            set_req.model_state.pose.orientation.x = orientation[0]
            set_req.model_state.pose.orientation.y = orientation[1]
            set_req.model_state.pose.orientation.z = orientation[2]
            set_req.model_state.pose.orientation.w = orientation[3]
            self.set_model_state_srv(set_req)
