from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ignition.markers import MarkerProvider
from link_bot_pycommon.base_services import Services
from moonshine.numpy_utils import remove_batch, numpify, dict_of_sequences_to_sequence_of_dicts


class ExperimentScenario:

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.simple_name()
        raise NotImplementedError()

    def simple_name(self):
        raise NotImplementedError()

    @staticmethod
    def sample_action(service_provider: Services, state, last_action, params, action_rng):
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
    def plot_state(ax, state: Dict, color, s: int, zorder: int, label: str):
        raise NotImplementedError()

    @staticmethod
    def plot_environment(ax, environment: Dict):
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
        raise NotImplementedError()

    @staticmethod
    def update_action_artist(artist, state, action):
        raise NotImplementedError()

    @staticmethod
    def update_artist(artist, state):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    @staticmethod
    def robot_name():
        raise NotImplementedError()

    @staticmethod
    def dynamics_loss_function(dataset_element, predictions):
        raise NotImplementedError()

    @staticmethod
    def dynamics_metrics_function(dataset_element, predictions):
        raise NotImplementedError()

    @staticmethod
    def integrate_dynamics(s_t, ds_t):
        raise NotImplementedError()

    @staticmethod
    def get_environment_from_example(example: Dict):
        raise NotImplementedError()

    @staticmethod
    def get_environment_from_start_states_dict(start_states: Dict):
        raise NotImplementedError()

    @classmethod
    def animate_predictions(cls, example_idx, dataset_element, predictions):
        predictions = remove_batch(predictions)
        predictions = numpify(dict_of_sequences_to_sequence_of_dicts(predictions))
        inputs, outputs = dataset_element
        actions = inputs['action']
        assert actions.shape[0] == 1
        actions = remove_batch(actions)
        outputs = remove_batch(outputs)
        inputs = numpify(remove_batch(inputs))
        actual = numpify(dict_of_sequences_to_sequence_of_dicts(outputs))
        fig = plt.figure()
        ax = plt.gca()
        prediction_artist = cls.plot_state(ax, predictions[0], 'g', zorder=3, s=10, label='prediction')
        actual_artist = cls.plot_state(ax, actual[0], '#00ff00', zorder=3, s=30, label='actual')
        action_artist = cls.plot_action(ax, actual[0], actions[0], color='c', s=30, zorder=4)
        environment = {
            'full_env/env': inputs['full_env/env'],
            'full_env/extent': inputs['full_env/extent'],
        }
        cls.plot_environment(ax, environment)
        ax.set_title("{}, t=0".format(example_idx))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.legend()

        n_states = len(actual)

        def update(t):
            cls.update_artist(prediction_artist, predictions[t])
            cls.update_artist(actual_artist, actual[t])
            ax.set_title("{}, t={}".format(example_idx, t))
            if t < n_states - 1:
                cls.update_action_artist(action_artist, actual[t], actions[t])

        anim = FuncAnimation(fig, update, interval=500, repeat=True, frames=n_states)
        return anim
