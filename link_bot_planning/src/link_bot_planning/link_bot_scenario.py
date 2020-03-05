from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ignition.markers import MarkerProvider
from link_bot_planning.experiment_scenario import ExperimentScenario


class LinkBotScenario(ExperimentScenario):

    @staticmethod
    def plot_state_simple(ax: plt.Axes,
                          state: Dict[str, np.ndarray],
                          color):
        link_bot_points = np.reshape(state['link_bot'], [-1, 2])
        x = link_bot_points[0, 0]
        y = link_bot_points[0, 1]
        ax.scatter(x, y, c=color, s=1)

    @staticmethod
    def plot_state(ax: plt.Axes,
                   state: Dict[str, np.ndarray],
                   color):
        link_bot_points = np.reshape(state['link_bot'], [-1, 2])
        xs = link_bot_points[:, 0]
        ys = link_bot_points[:, 1]
        ax.scatter(xs[0], ys[0], c=color, s=1)
        line = ax.plot(xs, ys, linewidth=1, c=color)[0]
        return line

    @staticmethod
    def distance_to_goal(
            state: Dict[str, np.ndarray],
            goal: np.ndarray):
        """
        Uses the first point in the link_bot subspace as the thing which we want to move to goal
        :param state: A dictionary of numpy arrays
        :param goal: Assumed to be a point in 2D
        :return:
        """
        link_bot_points = np.reshape(state['link_bot'], [-1, 2])
        tail_point = link_bot_points[0]
        distance = np.linalg.norm(tail_point - goal)
        return distance

    @staticmethod
    def distance_to_goal_differentiable(state, goal):
        link_bot_points = tf.reshape(state['link_bot'], [-1, 2])
        tail_point = link_bot_points[0]
        distance = tf.linalg.norm(tail_point - goal)
        return distance

    @staticmethod
    def distance(s1, s2):
        return np.linalg.norm(s1['link_bot'] - s2['link_bot'])

    @staticmethod
    def distance_differentiable(s1, s2):
        return tf.linalg.norm(s1['link_bot'] - s2['link_bot'])

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
        goal_points = np.reshape(link_bot_state, [-1, 2])
        goal_points -= goal_points[0]
        goal_points += goal
        goal_state = goal_points.flatten()
        return {
            'link_bot': goal_state
        }

    @staticmethod
    def plot_goal(ax, goal, color='g'):
        ax.scatter(goal[0], goal[1], s=50, c=color)

    @staticmethod
    def update_artist(artist, state):
        """ artist: Whatever was returned by plot_state """
        link_bot_points = np.reshape(state['link_bot'], [-1, 2])
        xs = link_bot_points[:, 0]
        ys = link_bot_points[:, 1]
        artist.set_data(xs, ys)

    @staticmethod
    def publish_state_marker(marker_provider: MarkerProvider, state):
        link_bot_points = np.reshape(state['link_bot'], [-1, 2])
        tail_point = link_bot_points[0]
        marker_provider.publish_marker(id=0, rgb=[1, 0, 0], scale=0.05, x=tail_point[0], y=tail_point[1])

    @staticmethod
    def publish_goal_marker(marker_provider: MarkerProvider, goal):
        marker_provider.publish_marker(id=0, rgb=[1, 0, 0], scale=0.05, x=goal[0], y=goal[1])

    @staticmethod
    def local_environment_center(state):
        link_bot_points = np.reshape(state['link_bot'], [-1, 2])
        head_point_where_gripper_is = link_bot_points[-1]
        return head_point_where_gripper_is
