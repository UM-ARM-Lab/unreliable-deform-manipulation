from typing import Dict

from ignition.markers import MarkerProvider


class ExperimentScenario:

    @staticmethod
    def movable_objects():
        raise NotImplementedError()

    @staticmethod
    def local_environment_center(state):
        raise NotImplementedError()

    @staticmethod
    def local_environment_center_differentiable(state):
        raise NotImplementedError()

    @staticmethod
    def plot_state_simple(ax, state, color, label=None, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def plot_state(ax, state: Dict, color, s: int, zorder: int):
        raise NotImplementedError()

    @staticmethod
    def plot_action(ax, state: Dict, action, color, s: int, zorder: int):
        raise NotImplementedError()

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
        pass

    @staticmethod
    def update_action_artist(artist, state, action):
        pass

    @staticmethod
    def update_artist(artist, state):
        pass

    def __repr__(self):
        raise NotImplementedError()
