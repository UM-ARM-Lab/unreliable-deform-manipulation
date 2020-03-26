from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ignition.markers import MarkerProvider
from link_bot_planning.experiment_scenario import ExperimentScenario


class TetherScenario(ExperimentScenario):

    @staticmethod
    def plot_state_simple(ax: plt.Axes,
                          state: Dict[str, np.ndarray],
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
                   zorder: int):
        point_robot = np.reshape(state['link_bot'], [2])
        x = point_robot[0]
        y = point_robot[1]
        scatt = ax.scatter(x, y, c=color, s=s, zorder=zorder)
        line = ax.plot(x, y, linewidth=1, c=color, zorder=zorder)[0]
        return line, scatt

    @staticmethod
    def distance_to_goal(state: Dict[str, np.ndarray],
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
    def update_artist(artist, state):
        """ artist: Whatever was returned by plot_state """
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
    @tf.function
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
