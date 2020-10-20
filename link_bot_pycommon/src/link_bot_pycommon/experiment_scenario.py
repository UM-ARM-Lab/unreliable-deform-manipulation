from typing import Dict, Optional

import numpy as np

import rospy
from arc_utilities.ros_helpers import TF2Wrapper
from geometry_msgs.msg import Vector3
from link_bot_data.link_bot_dataset_utils import add_predicted
from moonshine.moonshine_utils import numpify
from peter_msgs.srv import GetPosition3DRequest, Position3DEnableRequest, Position3DActionRequest
from std_msgs.msg import Int64, Float32


class ExperimentScenario:
    def __init__(self):
        self.time_viz_pub = rospy.Publisher("rviz_anim/time", Int64, queue_size=10, latch=True)
        self.traj_idx_viz_pub = rospy.Publisher("traj_idx_viz", Float32, queue_size=10, latch=True)
        self.tf = TF2Wrapper()

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
                      action_rng: np.random.RandomState,
                      environment: Dict,
                      state: Dict,
                      data_collection_params: Dict,
                      action_params: Dict,
                      stateless: Optional[bool] = False):
        raise NotImplementedError()

    def sample_action_sequences(self,
                                environment: Dict,
                                state: Dict,
                                data_collection_params: Dict,
                                action_params: Dict,
                                n_action_sequences: int,
                                action_sequence_length: int,
                                action_rng: np.random.RandomState):
        action_sequences = []

        for _ in range(n_action_sequences):
            action_sequence = self.sample_action_batch(environment=environment,
                                                       state=state,
                                                       data_collection_params=data_collection_params,
                                                       action_params=action_params,
                                                       batch_size=action_sequence_length,
                                                       action_rng=action_rng)
            action_sequences.append(action_sequence)
        return action_sequences

    def sample_action_batch(self,
                            environment: Dict,
                            state: Dict,
                            data_collection_params: Dict,
                            action_params: Dict,
                            batch_size: int,
                            action_rng: np.random.RandomState):
        action_sequence = []
        for __ in range(batch_size):
            action = self.sample_action(action_rng=action_rng,
                                        environment=environment,
                                        state=state,
                                        data_collection_params=data_collection_params,
                                        action_params=action_params,
                                        stateless=True)
            action_sequence.append(action)
        return action_sequence

    @staticmethod
    def local_environment_center_differentiable(state):
        raise NotImplementedError()

    def plot_goal_rviz(self, goal: Dict, goal_threshold: float, actually_at_goal: Optional[bool] = None):
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

    def sample_goal(self, environment: Dict, rng: np.random.RandomState, planner_params: Dict):
        raise NotImplementedError()

    @staticmethod
    def distance_to_goal(state, goal):
        raise NotImplementedError()

    def batch_full_distance(self, s1: Dict, s2: Dict):
        """ this is not the distance metric used in planning """
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

    @staticmethod
    def put_state_local_frame(state: Dict):
        raise NotImplementedError()

    @staticmethod
    def random_object_position(w: float, h: float, c: float, padding: float, rng: np.random.RandomState):
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

    def move_objects_randomly(self, env_rng, movable_objects_services, movable_objects, kinematic: bool,
                              timeout: float = 0.5):
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

    # TODO: the _keys_ for these "descriptions" should come from the data collection params file
    def states_description(self) -> Dict:
        raise NotImplementedError()

    @staticmethod
    def observations_description() -> Dict:
        raise NotImplementedError()

    @staticmethod
    def observation_features_description() -> Dict:
        raise NotImplementedError()

    @staticmethod
    def actions_description() -> Dict:
        raise NotImplementedError()

    @staticmethod
    def put_action_local_frame(state: Dict, action: Dict):
        raise NotImplementedError()

    def get_state(self):
        raise NotImplementedError()

    # unfortunately all these versions of these functions are slightly different
    # and trying to write it generically would be hard to do correctly and the result would be unreadable
    def index_predicted_state_time_batched(self, state, t):
        state_t = {}
        for feature_name in self.states_description().keys():
            if add_predicted(feature_name) in state:
                state_t[feature_name] = state[add_predicted(feature_name)][:, t]
        return state_t

    def index_observation_features_time_batched(self, observation_features, t):
        observation_features_t = {}
        for feature_name in self.observation_features_description().keys():
            if feature_name in observation_features:
                observation_features_t[feature_name] = observation_features[feature_name][:, t]
        return observation_features_t

    def index_observation_time_batched(self, observation, t):
        observation_t = {}
        for feature_name in self.observations_description().keys():
            if feature_name in observation:
                observation_t[feature_name] = observation[feature_name][:, t]
        return observation_t

    def index_state_time_batched(self, state, t):
        state_t = {}
        for feature_name in self.states_description().keys():
            if feature_name in state:
                state_t[feature_name] = state[feature_name][:, t]
        return state_t

    def index_action_time_batched(self, action, t):
        action_t = {}
        for feature_name in self.actions_description().keys():
            if feature_name in action:
                if t < action[feature_name].shape[0]:
                    action_t[feature_name] = action[feature_name][:, t]
                else:
                    action_t[feature_name] = action[feature_name][:, t - 1]
        return action_t

    def index_time_batched(self, e, t):
        e_t = {}
        all_keys = self.all_description_keys()
        for feature_name in all_keys:
            if feature_name in e:
                if t < e[feature_name].shape[0]:
                    e_t[feature_name] = e[feature_name][:, t]
                else:
                    e_t[feature_name] = e[feature_name][:, t - 1]
        return e_t

    @staticmethod
    def index_label_time_batched(example: Dict, t: int):
        if t == 0:
            # it makes no sense to have a label at t=0, labels are for transitions/sequences
            # the plotting function that consumes this should use None correctly
            return None
        return example['is_close'][:, t]

    def all_description_keys(self):
        all_keys = list(self.actions_description().keys()) \
                   + list(self.states_description().keys()) \
                   + list(self.observation_features_description().keys()) \
                   + list(self.observations_description().keys())
        return all_keys

    def randomization_initialization(self):
        raise NotImplementedError()

    def randomize_environment(self, env_rng: np.random.RandomState, objects_params: Dict, data_collection_params: Dict):
        raise NotImplementedError()

    def plot_traj_idx_rviz(self, traj_idx):
        msg = Float32()
        msg.data = traj_idx
        self.traj_idx_viz_pub.publish(msg)

    def plot_time_idx_rviz(self, time_idx):
        msg = Int64()
        msg.data = time_idx
        self.time_viz_pub.publish(msg)

    def dynamics_dataset_metadata(self):
        return {}

    def plot_transition_rviz(self, example: Dict, t):
        self.plot_environment_rviz(example)
        debugging_pred_state_t = numpify(self.index_predicted_state_time(example, t))
        self.plot_state_rviz(debugging_pred_state_t, label='predicted', color='b')
        # true state(not known to classifier!)
        debugging_true_state_t = numpify(self.index_state_time(example, t))
        self.plot_state_rviz(debugging_true_state_t, label='actual')
        debugging_action_t = numpify(self.index_action_time(example, t))
        self.plot_action_rviz(debugging_pred_state_t, debugging_action_t)
        label_t = self.index_label_time(example, t)
        self.plot_is_close(label_t)

    def get_environment(self, params: Dict, **kwargs):
        raise NotImplementedError()

    def on_before_data_collection(self, params: Dict):
        pass

    def trajopt_action_sequence_cost_differentiable(self, actions):
        return 0.0

    def trajopt_distance_to_goal_differentiable(self, final_state, goal):
        raise NotImplementedError()

    def trajopt_distance_differentiable(self, s1, s2):
        raise NotImplementedError()

    def get_excluded_models_for_env(self):
        raise NotImplementedError()

    def cfm_distance(self, z1, z2):
        raise NotImplementedError()

    def on_after_data_collection(self, params):
        pass


def sample_object_position(env_rng, xyz_range: Dict):
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
