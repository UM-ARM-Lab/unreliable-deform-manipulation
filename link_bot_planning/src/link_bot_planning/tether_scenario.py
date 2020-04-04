from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ignition.markers import MarkerProvider
from link_bot_data.visualization import plot_arrow, update_arrow
from link_bot_planning.experiment_scenario import ExperimentScenario
from link_bot_planning.params import CollectDynamicsParams
from link_bot_pycommon.base_services import Services
from moonshine.base_learned_dynamics_model import dynamics_loss_function, dynamics_points_metrics_function
from peter_msgs.msg import Action


class TetherScenario(ExperimentScenario):

    @staticmethod
    def random_delta_pos(action_rng, max_delta_pos):
        delta_pos = action_rng.uniform(0, max_delta_pos)
        direction = action_rng.uniform(-np.pi, np.pi)
        dx = np.cos(direction) * delta_pos
        dy = np.sin(direction) * delta_pos
        return dx, dy

    @staticmethod
    def sample_action(service_provider: Services,
                      state,
                      last_action: Action,
                      params: CollectDynamicsParams,
                      goal_w_m,
                      goal_h_m,
                      action_rng):
        max_delta_pos = service_provider.get_max_speed() * params.dt
        new_action = Action()
        while True:
            # sample the previous action with 80% probability
            # we implicit use a dynamics model for the gripper here, which in this case is identity linear dynamics
            if last_action is not None and action_rng.uniform(0, 1) < 0.80:
                dx = last_action.action[0]
                dy = last_action.action[1]
            else:
                dx, dy = TetherScenario.random_delta_pos(action_rng, max_delta_pos)

            half_w = goal_w_m / 2
            half_h = goal_h_m / 2
            if -half_w <= state['gripper'][0] + dx <= half_w and -half_h <= state['gripper'][1] + dy <= half_h:
                break

        new_action.action = [dx, dy]
        new_action.max_time_per_step = params.dt
        return new_action

    @staticmethod
    def plot_state_simple(ax: plt.Axes,
                          state: Dict,
                          color,
                          label=None,
                          **kwargs):
        point_robot = np.reshape(state['link_bot'], [2])
        x = point_robot[0]
        y = point_robot[1]
        ax.scatter(x, y, c=color, label=label, **kwargs)

    @staticmethod
    def plot_state(ax: plt.Axes,
                   state: Dict,
                   color,
                   s: int,
                   zorder: int,
                   label: Optional[str] = None):
        if 'tether' in state.keys():
            tether = np.reshape(state['tether'], [-1, 2])
            xs = tether[:, 0]
            ys = tether[:, 1]
            scatt = ax.scatter(xs, ys, c=color, s=s, zorder=zorder, label=label)
            line = ax.plot(xs, ys, linewidth=1, c=color, zorder=zorder)[0]
            return line, scatt
        elif 'link_bot' in state.keys():
            point_robot = np.reshape(state['link_bot'], [2])
            x = point_robot[0]
            y = point_robot[1]
            scatt = ax.scatter(x, y, c=color, s=s, zorder=zorder, label=label)
            line = ax.plot(x, y, linewidth=1, c=color, zorder=zorder)[0]
            return line, scatt
        elif 'gripper' in state.keys():
            point_robot = np.reshape(state['gripper'], [2])
            x = point_robot[0]
            y = point_robot[1]
            scatt = ax.scatter(x, y, c=color, s=s, zorder=zorder, label=label)
            line = ax.plot(x, y, linewidth=1, c=color, zorder=zorder)[0]
            return line, scatt

    @staticmethod
    def plot_action(ax: plt.Axes,
                    state: Dict,
                    action,
                    color,
                    s: int,
                    zorder: int):
        if 'link_bot' in state.keys():
            point_robot = np.reshape(state['link_bot'], [2])
            # we draw our own arrows because quiver cannot be animated
            artist = plot_arrow(ax, point_robot[0], point_robot[1], action[0], action[1], zorder=zorder, linewidth=1, color=color)
            return artist
        elif 'gripper' in state.keys():
            point_robot = np.reshape(state['link_bot'], [2])
            # we draw our own arrows because quiver cannot be animated
            artist = plot_arrow(ax, point_robot[0], point_robot[1], action[0], action[1], zorder=zorder, linewidth=1, color=color)
            return artist

    @staticmethod
    def distance_to_goal(state: Dict,
                         goal: np.ndarray):
        """
        Uses the first point in the link_bot subspace as the thing which we want to move to goal
        :param state: A dictionary of numpy arrays
        :param goal: Assumed to be a point in 2D
        :return:
        """
        point_robot = np.reshape(state['link_bot'], [2])
        distance = np.linalg.norm(point_robot - goal)
        return distance

    @staticmethod
    def distance_to_goal_differentiable(state, goal):
        point_robot = tf.reshape(state['link_bot'], [2])
        distance = tf.linalg.norm(point_robot - goal)
        return distance

    @staticmethod
    def distance(s1, s2):
        point_robot1 = np.reshape(s1['link_bot'], [2])
        point_robot2 = np.reshape(s2['link_bot'], [2])
        return np.linalg.norm(point_robot1 - point_robot2)

    @staticmethod
    def distance_differentiable(s1, s2):
        point_robot1 = tf.reshape(s1['link_bot'], [2])
        point_robot2 = tf.reshape(s2['link_bot'], [2])
        return tf.linalg.norm(point_robot1 - point_robot2)

    @staticmethod
    def get_subspace_weight(subspace_name: str):
        if subspace_name == 'link_bot':
            return 1.0
        elif subspace_name == 'tether':
            return 1.0
        else:
            raise NotImplementedError("invalid subspace_name {}".format(subspace_name))

    @staticmethod
    def sample_goal(state, goal):
        del state  ## unused
        goal_state = np.reshape(goal, [2])
        return {
            'link_bot': goal_state
        }

    @staticmethod
    def plot_goal(ax, goal, color='g', label=None, **kwargs):
        ax.scatter(goal[0], goal[1], c=color, label=label, **kwargs)

    @staticmethod
    def update_action_artist(artist, state, action):
        """ artist: Whatever was returned by plot_state """
        point_robot = np.reshape(state['link_bot'], [2])
        update_arrow(artist, point_robot[0], point_robot[1], action[0], action[1])

    @staticmethod
    def update_artist(artist, state):
        """ artist: Whatever was returned by plot_state """
        if 'tether' in state.keys():
            line, scatt = artist

            tether = np.reshape(state['tether'], [-1, 2])
            xs = tether[:, 0]
            ys = tether[:, 1]
            scatt.set_offsets(tether)
            line.set_data(xs, ys)
        else:
            line, scatt = artist
            link_bot_points = np.reshape(state['link_bot'], [-1, 2])
            xs = link_bot_points[:, 0]
            ys = link_bot_points[:, 1]
            line.set_data(xs, ys)
            scatt.set_offsets(link_bot_points[0])

    @staticmethod
    def publish_state_marker(marker_provider: MarkerProvider, state):
        point_robot = np.reshape(state['link_bot'], [2])
        marker_provider.publish_marker(id=0, rgb=[1, 0, 0], scale=0.05, x=point_robot[0], y=point_robot[1])

    @staticmethod
    def publish_goal_marker(marker_provider: MarkerProvider, goal, size: float):
        marker_provider.publish_marker(id=0, rgb=[1, 0, 0], scale=size, x=goal[0], y=goal[1])

    @staticmethod
    def local_environment_center(state):
        point_robot = state['link_bot']
        return point_robot

    @staticmethod
    # @tf.function
    def local_environment_center_differentiable(state):
        """
        :param state: Dict of batched states
        :return:
        """
        b = int(state['link_bot'].shape[0])
        batched_point_robot = tf.reshape(state['link_bot'], [b, 2])
        return batched_point_robot

    def __repr__(self):
        return "tether"

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
