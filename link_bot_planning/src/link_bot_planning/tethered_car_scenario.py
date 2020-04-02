from typing import Dict

import numpy as np
import tensorflow as tf

from ignition.markers import MarkerProvider
from link_bot_data.visualization import plot_arrow, update_arrow
from link_bot_planning.experiment_scenario import ExperimentScenario
from link_bot_pycommon.base_services import Services
from link_bot_pycommon.link_bot_pycommon import wrap_angle
from moonshine.base_learned_dynamics_model import dynamics_loss_function
from peter_msgs.msg import Action

speed_arrow_scale = 0.05


class TetheredCarScenario(ExperimentScenario):

    @staticmethod
    def sample_action(service_provider: Services, state, last_action, params, goal_w_m, goal_h_m, action_rng):
        max_speed = service_provider.get_max_speed() * params.dt
        new_action = Action()
        if last_action is not None and action_rng.uniform(0, 1) < 0.80:
            left_wheel_speed = last_action.action[0]
            right_wheel_speed = last_action.action[1]
        else:
            left_wheel_speed = action_rng.uniform(-max_speed, max_speed)
            right_wheel_speed = action_rng.uniform(-max_speed, max_speed)

        new_action.action = [left_wheel_speed, right_wheel_speed]
        new_action.max_time_per_step = params.dt
        return new_action

    @staticmethod
    def local_environment_center(state):
        raise NotImplementedError()

    @staticmethod
    def local_environment_center_differentiable(state):
        raise NotImplementedError()

    @staticmethod
    def plot_state_simple(ax, state, color, label=None, **kwargs):
        # this doesn't visualize yaw or yaw rate
        return ax.scatter(state['car'][0], state['car'][1], c=color, label=label, **kwargs)

    @staticmethod
    def plot_state(ax, state: Dict, color, s: int, zorder: int):
        # this doesn't visualize yaw or yaw rate
        x = state['car'][0]
        y = state['car'][1]
        u = state['car'][3]
        v = state['car'][4]
        scatt = ax.scatter(state['car'][0], state['car'][1], c=color, s=s, zorder=zorder)
        arrow_artist = plot_arrow(ax, x, y, u, v, color=color, zorder=zorder)
        return [scatt, arrow_artist]

    @staticmethod
    def plot_action(ax, state: Dict, action, color, s: int, zorder: int):
        x = state['car'][0]
        y = state['car'][1]
        yaw = state['car'][2]
        left_wheel_x = x - np.sin(yaw) * 0.05
        left_wheel_y = y + np.cos(yaw) * 0.05
        right_wheel_x = x - np.sin(yaw) * -0.05
        right_wheel_y = y + np.cos(yaw) * -0.05
        left_speed = action[0] * speed_arrow_scale
        left_dx = left_speed * np.cos(yaw)
        left_dy = left_speed * np.sin(yaw)
        right_speed = action[1] * speed_arrow_scale
        right_dx = right_speed * np.cos(yaw)
        right_dy = right_speed * np.sin(yaw)
        left_arrow_artist = plot_arrow(ax, left_wheel_x, left_wheel_y, left_dx, left_dy, color=color, zorder=zorder)
        right_arrow_artist = plot_arrow(ax, right_wheel_x, right_wheel_y, right_dx, right_dy, color=color, zorder=zorder)
        return [left_arrow_artist, right_arrow_artist]

    @staticmethod
    def plot_goal(ax, goal, color, label=None, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def publish_goal_marker(marker_provider: MarkerProvider, goal, size: float):
        raise NotImplementedError()

    @staticmethod
    def publish_state_marker(marker_provider: MarkerProvider, state):
        raise NotImplementedError()

    @staticmethod
    def distance_to_goal(state, goal):
        raise NotImplementedError()

    @staticmethod
    def distance_to_goal_differentiable(state, goal):
        raise NotImplementedError()

    @staticmethod
    def distance(s1, s2):
        raise NotImplementedError()

    @staticmethod
    def distance_differentiable(s1, s2):
        raise NotImplementedError()

    @staticmethod
    def get_subspace_weight(subspace_name: str):
        raise NotImplementedError()

    @staticmethod
    def sample_goal(state, goal):
        raise NotImplementedError()

    @staticmethod
    def update_action_artist(artist, state, action):
        left_arrow_artist, right_arrow_artist = artist
        x = state['car'][0]
        y = state['car'][1]
        yaw = state['car'][2]
        left_wheel_x = x - np.sin(yaw) * 0.05
        left_wheel_y = y + np.cos(yaw) * 0.05
        right_wheel_x = x - np.sin(yaw) * -0.05
        right_wheel_y = y + np.cos(yaw) * -0.05
        left_speed = action[0] * speed_arrow_scale
        left_dx = left_speed * np.cos(yaw)
        left_dy = left_speed * np.sin(yaw)
        right_speed = action[1] * speed_arrow_scale
        right_dx = right_speed * np.cos(yaw)
        right_dy = right_speed * np.sin(yaw)
        update_arrow(left_arrow_artist, left_wheel_x, left_wheel_y, left_dx, left_dy)
        update_arrow(right_arrow_artist, right_wheel_x, right_wheel_y, right_dx, right_dy)

    @staticmethod
    def update_artist(artist, state):
        [scatt, arrow_artist] = artist
        x = state['car'][0]
        y = state['car'][1]
        u = state['car'][3]
        v = state['car'][4]
        update_arrow(arrow_artist, x, y, u, v)
        scatt.set_offsets([x, y])

    def __repr__(self):
        return "tethered-car"

    @staticmethod
    def robot_name():
        return "car"

    @staticmethod
    @tf.function
    def dynamics_loss_function(dataset_element, predictions):
        return dynamics_loss_function(dataset_element, predictions)

    @staticmethod
    @tf.function
    def dynamics_metrics_function(dataset_element, predictions):
        input_data, output_data = dataset_element
        metrics = {}
        for state_key, pred_state in predictions.items():
            true_state = output_data[state_key]
            true_pos = tf.reshape(true_state[:, :, 0:2], [true_state.shape[0], true_state.shape[1], 2])
            pred_pos = tf.reshape(pred_state[:, :, 0:2], [pred_state.shape[0], pred_state.shape[1], 2])
            position_errors = tf.linalg.norm(pred_pos - true_pos, axis=2)
            mean_position_error = tf.reduce_mean(position_errors)
            final_position_error = tf.reduce_mean(position_errors[:, -1])
            angle_errors = tf.math.abs(true_state[:, :, 2] - pred_state[:, :, 2])
            mean_angle_error = tf.reduce_mean(angle_errors)
            final_angle_error = tf.reduce_mean(angle_errors[:, -1])

            metrics['{} mean position error'.format(state_key)] = mean_position_error
            metrics['{} final position error'.format(state_key)] = final_position_error
            metrics['{} mean angle error'.format(state_key)] = mean_angle_error
            metrics['{} final angle error'.format(state_key)] = final_angle_error
        return metrics

    @staticmethod
    # @tf.function
    def integrate_dynamics(s_t, ds_t):
        s_t_plus_1 = s_t + ds_t
        mask = tf.constant([0, 0, 1, 0, 0, 0], dtype=tf.float32)
        inv_mask = 1.0 - mask
        s_t_plus_1 = s_t_plus_1 * inv_mask + ((s_t_plus_1 + np.pi) % (2 * np.pi) - np.pi) * mask
        return s_t_plus_1
