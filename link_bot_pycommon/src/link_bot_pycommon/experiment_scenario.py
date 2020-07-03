import multiprocessing
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import numpy as np

import rospy
from geometry_msgs.msg import Vector3
from link_bot_data.link_bot_dataset_utils import add_predicted
from link_bot_data.visualization import plot_extents
from link_bot_pycommon.animation_player import Player
from link_bot_pycommon.pycommon import trim_reconverging
from moonshine.moonshine_utils import remove_batch, numpify, dict_of_sequences_to_sequence_of_dicts_tf
from peter_msgs.srv import GetPosition3DRequest, Position3DEnableRequest, Position3DActionRequest


class ExperimentScenario:
    def __init__(self, params: Dict):
        self.params = params

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.simple_name()
        raise NotImplementedError()

    @staticmethod
    def simple_name():
        raise NotImplementedError()

    def execute_action(self, action: Dict):
        raise NotImplementedError()

    def sample_action(self,
                      environment: Dict,
                      state: Dict,
                      params: Dict, action_rng):
        raise NotImplementedError()

    def sample_actions(self,
                       environment: Dict,
                       start_state: Dict,
                       params: Dict,
                       n_action_sequences: int,
                       action_sequence_length: int,
                       rng: np.random.RandomState):
        action_sequences = []

        for _ in range(n_action_sequences):
            action_sequence = []
            for __ in range(action_sequence_length):
                action = self.sample_action(environment=environment,
                                            state=start_state,
                                            params=params,
                                            action_rng=rng)
                action_sequence.append(action)
            action_sequences.append(action_sequence)
        return action_sequences

    @staticmethod
    def local_environment_center(state):
        raise NotImplementedError()

    @staticmethod
    def local_environment_center_differentiable(state):
        raise NotImplementedError()

    @staticmethod
    def plot_state_simple(ax, state: Dict, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def plot_state(ax, state: Dict, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def plot_environment(ax, environment: Dict):
        raise NotImplementedError()

    @staticmethod
    def plot_action(ax, state: Dict, action, color, s: int, zorder: int, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def plot_goal(ax, goal, color, label=None, **kwargs):
        raise NotImplementedError()

    def plot_environment_rviz(self, data: Dict):
        raise NotImplementedError()

    def plot_state_rviz(self, data: Dict, label: str, **kwargs):
        raise NotImplementedError()

    def plot_action_rviz(self, state: Dict, action: Dict, **kwargs):
        raise NotImplementedError()

    def plot_is_close(self, label_t):
        raise NotImplementedError()

    def animate_rviz(self, environment, actual_states, predicted_states, actions, labels, accept_probabilities):
        raise NotImplementedError()

    @staticmethod
    def state_to_points_for_cc(state: Dict):
        raise NotImplementedError()

    @staticmethod
    def state_to_gripper_position(state: Dict):
        raise NotImplementedError()

    @staticmethod
    def publish_goal_marker(goal, size: float):
        raise NotImplementedError()

    def sample_goal(self, environment: Dict, rng: np.random.RandomState):
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
    def get_environment_from_state_dict(start_states: Dict):
        raise NotImplementedError()

    @classmethod
    def animate_predictions_from_classifier_dataset(cls,
                                                    state_keys: List[str],
                                                    example_idx: int,
                                                    dataset_element: Dict,
                                                    trim: Optional[bool] = False,
                                                    accept_probabilities: Optional = None,
                                                    fps: Optional[int] = 1):
        is_close = dataset_element['is_close'].numpy()

        if trim:
            start_idx, end_idx = trim_reconverging(is_close)
        else:
            start_idx = 0
            end_idx = len(is_close)

        predictions = {}
        actual = {}
        for state_key in state_keys:
            predictions[state_key] = dataset_element[add_predicted(state_key)]
            actual[state_key] = dataset_element[state_key]

        predictions = numpify(dict_of_sequences_to_sequence_of_dicts_tf(predictions))
        actual = numpify(dict_of_sequences_to_sequence_of_dicts_tf(actual))
        actions = dataset_element['action']
        environment = numpify({
            'full_env/env': dataset_element['full_env/env'],
            'full_env/extent': dataset_element['full_env/extent'],
        })

        if 'mask' in dataset_element:
            mask = np.concatenate([[1], dataset_element['mask'].numpy().astype(np.int64)], axis=0)
            show_previous_action = True
        else:
            mask = None
            show_previous_action = False

        return cls.animate_predictions(environment=environment,
                                       actions=actions[start_idx:end_idx],
                                       actual=actual[start_idx:end_idx],
                                       predictions=predictions[start_idx:end_idx],
                                       example_idx=example_idx,
                                       labels=is_close[start_idx:end_idx],
                                       mask=mask,
                                       accept_probabilities=accept_probabilities,
                                       show_previous_action=show_previous_action,
                                       fps=fps)

    @classmethod
    def animation_data_from_dynamics_dataset(cls, dataset_element, predictions, labels=None, start_idx=0, end_idx=-1):
        predictions = remove_batch(predictions)
        predictions = numpify(dict_of_sequences_to_sequence_of_dicts_tf(predictions))
        inputs, outputs = dataset_element
        actions = inputs['action']
        assert actions.shape[0] == 1
        actions = remove_batch(actions)
        outputs = remove_batch(outputs)
        inputs = numpify(remove_batch(inputs))
        actual = numpify(dict_of_sequences_to_sequence_of_dicts_tf(outputs))
        extent = inputs['full_env/extent']
        environment = {
            'full_env/env': inputs['full_env/env'],
            'full_env/extent': extent,
        }

        labels = labels[start_idx:end_idx] if labels is not None else None
        return environment, actions[start_idx:end_idx], actual[start_idx:end_idx], predictions[start_idx:end_idx], labels

    @classmethod
    def animate_predictions_from_dynamics_dataset(cls,
                                                  example_idx,
                                                  dataset_element,
                                                  predictions,
                                                  labels=None,
                                                  start_idx=0,
                                                  end_idx=-1,
                                                  accept_probabilities: Optional = None,
                                                  fps: Optional[float] = 1):

        animation_data = cls.animation_data_from_dynamics_dataset(dataset_element=dataset_element,
                                                                  predictions=predictions,
                                                                  labels=labels,
                                                                  start_idx=start_idx,
                                                                  end_idx=end_idx)
        return cls.animate_predictions(*animation_data,
                                       example_idx=example_idx,
                                       accept_probabilities=accept_probabilities,
                                       fps=fps)

    @classmethod
    def animate_predictions(cls,
                            environment,
                            actions,
                            actual,
                            predictions: Optional,
                            labels: Optional = None,
                            example_idx: Optional = None,
                            accept_probabilities: Optional = None,
                            mask: Optional = None,
                            show_previous_action: bool = False,
                            fps: Optional[float] = 1):
        fig = plt.figure()
        ax = plt.gca()
        update, frames = cls.animate_predictions_on_axes(ax=ax,
                                                         environment=environment,
                                                         actions=actions,
                                                         actual=actual,
                                                         predictions=predictions,
                                                         example_idx=example_idx,
                                                         labels=labels,
                                                         accept_probabilities=accept_probabilities,
                                                         mask=mask,
                                                         show_previous_action=show_previous_action,
                                                         )

        plt.legend()
        anim = Player(fig, update, max_index=frames, interval=1000 / fps, repeat=True)
        return anim

    @classmethod
    def animate_predictions_on_axes(cls,
                                    ax,
                                    environment,
                                    actions,
                                    actual,
                                    predictions: Optional,
                                    labels: Optional = None,
                                    example_idx: Optional = None,
                                    accept_probabilities: Optional = None,
                                    mask: Optional = None,
                                    prediction_label_name: Optional = 'prediction',
                                    prediction_color: Optional = 'g',
                                    show_previous_action: bool = False,
                                    ):
        prediction_artist = None
        if predictions is not None:
            prediction_artist = cls.plot_state(ax,
                                               predictions[0],
                                               color=prediction_color,
                                               zorder=3,
                                               s=2,
                                               label=prediction_label_name,
                                               linewidth=1)
        actual_artist = cls.plot_state(ax, actual[0], color='#00ff00', zorder=3,
                                       s=2, label='actual', alpha=0.6, linewidth=1)
        if show_previous_action:
            prev_actual_artist = cls.plot_state(ax, actual[0], color='#aaaa00', zorder=3, s=2, label='actual', alpha=0.6,
                                                linewidth=1)
        if actions is not None:
            action_artist = cls.plot_action(ax, actual[0], actions[0], color='c', s=2, zorder=4, linewidth=1)
        cls.plot_environment(ax, environment)
        if labels is not None:
            extent = environment['full_env/extent'] * 1.05
            offset_extent = extent * 1.025
            label_line = plot_extents(ax=ax, extent=extent, color='k', zorder=2, alpha=0.5)
            classification_line = plot_extents(ax=ax, extent=offset_extent, color='k', zorder=3, alpha=0.5)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        n_frames = len(actual)
        if mask is not None:
            n_frames = np.count_nonzero(mask)

        def update(t):
            if mask is not None:
                valid_indices = np.where(mask)[0]
                t = valid_indices[t]
            if predictions is not None:
                cls.update_artist(prediction_artist, predictions[t])
            cls.update_artist(actual_artist, actual[t])
            if example_idx is None:
                title_t = f"t={t}"
            else:
                title_t = f"example {example_idx}, t={t}"
            if t > 0:
                if labels is not None:
                    title_t += f" label={labels[t]}"
                    label_color = 'r' if labels[t] == 0 else 'g'
                    label_line.set_color(label_color)
                if accept_probabilities is not None:
                    # -1 because the classifier doesn't output a decision for t=0
                    accept_probability = accept_probabilities[t - 1]
                    # TODO: use threshold from model hparams
                    line_color = 'r' if accept_probability < 0.5 else 'g'
                    classification_line.set_color(line_color)
                    title_t += f" accept={accept_probability:.3f}"
            else:
                if accept_probabilities is not None:
                    title_t += " label=    accept=     "
                elif labels is not None:
                    title_t += " label=    "
                    label_line.set_color('k')
                    classification_line.set_color('k')
            ax.set_title(title_t, fontproperties='monospace')

            if show_previous_action:
                if t > 0 and actions is not None:
                    cls.update_artist(prev_actual_artist, actual[t - 1])
                    cls.update_action_artist(action_artist, actual[t - 1], actions[t - 1])
            else:
                if t < n_frames - 1 and actions is not None:
                    cls.update_action_artist(action_artist, actual[t], actions[t])

        return update, n_frames

    @staticmethod
    def put_state_local_frame(state: Dict):
        raise NotImplementedError()

    @staticmethod
    def random_object_position(w: float, h: float, c: float, padding: float, rng: np.random.RandomState) -> Dict:
        xyz_range = {
            'x': [-w / 2 + padding, w / 2 - padding],
            'y': [-h / 2 + padding, h / 2 - padding],
            'z': [-c / 2 + padding, c / 2 - padding],
        }
        return sample_object_position(rng, xyz_range)

    @staticmethod
    def get_movable_object_positions(movable_objects_services: Dict):
        positions = {}
        for object_name, services in movable_objects_services.items():
            position_response = services['get_position'](GetPosition3DRequest())
            positions[object_name] = position_response
        return positions

    def move_objects_randomly(self, env_rng, movable_objects_services, movable_objects, kinematic: bool, timeout: float = 0.5):
        random_object_positions = sample_object_positions(env_rng, movable_objects)
        if kinematic:
            raise NotImplementedError()
        else:
            ExperimentScenario.move_objects(movable_objects_services, random_object_positions, timeout)

    @staticmethod
    def move_objects_to_positions(movable_objects_services: Dict, object_positions: Dict, timeout: float = 0.5):
        object_positions = {}
        for name, (x, y) in object_positions.items():
            position = Vector3()
            position.x = x
            position.y = y
            object_positions[name] = position
        return ExperimentScenario.move_objects(movable_objects_services, object_positions, timeout)

    @staticmethod
    def set_objects(movable_objects_services: Dict, object_positions: Dict, timeout: float):
        for name, position in object_positions.items():
            services = movable_objects_services[name]
            ExperimentScenario.call_set(services, name, position)

    @staticmethod
    def move_objects(movable_objects_services: Dict, object_positions: Dict, timeout: float):
        for name, position in object_positions.items():
            services = movable_objects_services[name]
            ExperimentScenario.call_move(services, name, position, timeout)

        # disable controller so objects can move around
        for object_name, _ in object_positions.items():
            movable_object_services = movable_objects_services[object_name]
            enable_req = Position3DEnableRequest()
            enable_req.enable = False
            movable_object_services['enable'](enable_req)

    @staticmethod
    def call_set(movable_object_services, object_name, position):
        set_action_req = Position3DActionRequest()
        set_action_req.position = position
        movable_object_services['set'](set_action_req)

    @staticmethod
    def call_move(movable_object_services, object_name, position, timeout):
        move_action_req = Position3DActionRequest()
        move_action_req.position = position
        move_action_req.timeout = timeout
        movable_object_services['move'](move_action_req)

    @staticmethod
    def states_description() -> Dict:
        raise NotImplementedError()

    @staticmethod
    def actions_description() -> Dict:
        raise NotImplementedError()

    @staticmethod
    def put_action_local_frame(state: Dict, action: Dict):
        raise NotImplementedError()

    @staticmethod
    def teleport_to_state(state: Dict):
        raise NotImplementedError()

    def get_state(self):
        raise NotImplementedError()

    def safety_policy(self, previous_state: Dict, new_state: Dict, environment: Dict):
        raise NotImplementedError()

    @staticmethod
    def index_state_time(state: Dict, t: int):
        raise NotImplementedError()

    @staticmethod
    def index_predicted_state_time(state: Dict, t: int):
        raise NotImplementedError()

    @staticmethod
    def index_action_time(action: Dict, t: int):
        raise NotImplementedError()

    @staticmethod
    def index_label_time(example: Dict, t: int):
        raise NotImplementedError()


def sample_object_position(env_rng, xyz_range: Dict) -> Dict:
    x_range = xyz_range['x']
    y_range = xyz_range['y']
    z_range = xyz_range['z']
    position = Vector3()
    position.x = env_rng.uniform(*x_range)
    position.y = env_rng.uniform(*y_range)
    position.z = env_rng.uniform(*z_range)
    return position


def sample_object_positions(env_rng, movable_objects: Dict) -> Dict[str, Dict]:
    random_object_positions = {name: sample_object_position(
        env_rng, xyz_range) for name, xyz_range in movable_objects.items()}
    return random_object_positions
