from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation

from ignition.markers import MarkerProvider
from link_bot_data.link_bot_dataset_utils import add_planned
from link_bot_data.visualization import plot_arrow, update_arrow
from link_bot_planning.experiment_scenario import ExperimentScenario
from link_bot_planning.params import CollectDynamicsParams
from link_bot_pycommon.base_services import Services
from moonshine.base_learned_dynamics_model import dynamics_loss_function, dynamics_points_metrics_function
from moonshine.numpy_utils import remove_batch, numpify, dict_of_sequences_to_sequence_of_dicts
from peter_msgs.msg import Action


class LinkBotScenario(ExperimentScenario):

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
                dx, dy = LinkBotScenario.random_delta_pos(action_rng, max_delta_pos)

            half_w = params.goal_w_m / 2
            half_h = params.goal_h_m / 2
            if -half_w <= state['gripper'][0] + dx <= half_w and -half_h <= state['gripper'][1] + dy <= half_h:
                break

        new_action.action = [dx, dy]
        new_action.max_time_per_step = params.dt
        return new_action

    @staticmethod
    def plot_state_simple(ax: plt.Axes,
                          state: Dict[str, np.ndarray],
                          color,
                          label=None,
                          **kwargs):
        link_bot_points = np.reshape(state['link_bot'], [-1, 2])
        x = link_bot_points[0, 0]
        y = link_bot_points[0, 1]
        ax.scatter(x, y, c=color, label=label, **kwargs)

    @staticmethod
    def plot_state(ax: plt.Axes,
                   state: Dict,
                   color,
                   s: int,
                   zorder: int,
                   label: Optional[str] = None):
        link_bot_points = np.reshape(state['link_bot'], [-1, 2])
        xs = link_bot_points[:, 0]
        ys = link_bot_points[:, 1]
        scatt = ax.scatter(xs[0], ys[0], c=color, s=s, zorder=zorder)
        line = ax.plot(xs, ys, linewidth=4, c=color, zorder=zorder, label=label)[0]
        return line, scatt

    @staticmethod
    def plot_action(ax, state: Dict, action, color, s: int, zorder: int):
        link_bot_points = np.reshape(state['link_bot'], [-1, 2])
        artist = plot_arrow(ax, link_bot_points[-2, 0], link_bot_points[-1, 1], action[0], action[1], zorder=zorder, linewidth=1,
                            color=color)
        return artist

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
        # NOTE: using R^22 distance mangles the rope shape more, so we don't use it.
        link_bot_points1 = np.reshape(s1['link_bot'], [-1, 2])
        tail_point1 = link_bot_points1[0]
        link_bot_points2 = np.reshape(s2['link_bot'], [-1, 2])
        tail_point2 = link_bot_points2[0]
        return np.linalg.norm(tail_point1 - tail_point2)

    @staticmethod
    def distance_differentiable(s1, s2):
        # NOTE: using R^22 distance mangles the rope shape more, so we don't use it.
        link_bot_points1 = tf.reshape(s1['link_bot'], [-1, 2])
        tail_point1 = link_bot_points1[0]
        link_bot_points2 = tf.reshape(s2['link_bot'], [-1, 2])
        tail_point2 = link_bot_points2[0]
        return tf.linalg.norm(tail_point1 - tail_point2)

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
    def plot_goal(ax, goal, color='g', label=None, **kwargs):
        ax.scatter(goal[0], goal[1], c=color, label=label, **kwargs)

    @staticmethod
    def plot_environment(ax, environment: Dict):
        occupancy = environment['full_env/env']
        extent = environment['full_env/extent']
        ax.imshow(np.flipud(occupancy), extent=extent, cmap='Greys')

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
    def update_action_artist(artist, state, action):
        """ artist: Whatever was returned by plot_state """
        link_bot_points = np.reshape(state['link_bot'], [-1, 2])
        update_arrow(artist, link_bot_points[-2, 0], link_bot_points[-1, 1], action[0], action[1])

    @staticmethod
    def publish_state_marker(marker_provider: MarkerProvider, state):
        link_bot_points = np.reshape(state['link_bot'], [-1, 2])
        tail_point = link_bot_points[0]
        marker_provider.publish_marker(id=0, rgb=[1, 0, 0], scale=0.05, x=tail_point[0], y=tail_point[1])

    @staticmethod
    def publish_goal_marker(marker_provider: MarkerProvider, goal, size: float):
        marker_provider.publish_marker(id=0, rgb=[1, 0, 0], scale=size, x=goal[0], y=goal[1])

    @staticmethod
    def local_environment_center(state):
        link_bot_points = tf.reshape(state['link_bot'], [-1, 2])
        head_point_where_gripper_is = link_bot_points[-1]
        return head_point_where_gripper_is

    @staticmethod
    @tf.function
    def local_environment_center_differentiable(state):
        """
        :param state: Dict of batched states
        :return:
        """
        if 'link_bot' in state:
            link_bot_state = state['link_bot']
        elif add_planned('link_bot') in state:
            link_bot_state = state[add_planned('link_bot')]
        b = int(link_bot_state.shape[0])
        link_bot_points = tf.reshape(link_bot_state, [b, -1, 2])
        head_point_where_gripper_is = link_bot_points[:, -1]
        return head_point_where_gripper_is

    def __repr__(self):
        return "Rope Manipulation"

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
    def get_environment_from_start_states_dict(start_states: Dict):
        return {}

    @staticmethod
    def get_environment_from_example(example: Dict):
        raise {}

