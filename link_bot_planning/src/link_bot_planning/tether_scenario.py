from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from link_bot_data.visualization import plot_rope_configuration
from link_bot_planning.experiment_scenario import ExperimentScenario


class TetherScenario(ExperimentScenario):

    @staticmethod
    def plot_state(ax: plt.Axes,
                   state: Dict[str, np.ndarray]):
        link_bot_points = np.reshape(state['link_bot'][0], [-1, 2])
        plot_rope_configuration(ax, link_bot_points)

    @staticmethod
    def distance_to_goal(state: Dict[str, np.ndarray],
                         goal: np.ndarray):
        """
        Uses the first point in the link_bot subspace as the thing which we want to move to goal
        :param ax:
        :param state: A dictionary of numpy arrays
        :param goal: Assumed to be a point in 2D
        :return:
        """
        link_bot_points = np.reshape(state['link_bot'][0], [-1, 2])
        tail_point = link_bot_points[0]
        distance = np.linalg.norm(tail_point - goal)
        return distance

    @staticmethod
    def get_subspace_weight(subspace_name: str):
        if subspace_name == 'link_bot':
            return 1.0
        elif subspace_name == 'tether':
            return 0.0
        else:
            raise NotImplementedError()

    def __repr__(self):
        return "tether"
