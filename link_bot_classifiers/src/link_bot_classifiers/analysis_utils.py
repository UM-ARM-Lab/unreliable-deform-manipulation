import numpy as np
import tensorflow as tf

from link_bot_classifiers import classifier_utils
from link_bot_planning.plan_and_execute import execute_actions
from link_bot_pycommon.pycommon import make_dict_tf_float32
from link_bot_pycommon.ros_pycommon import get_states_dict, make_movable_object_services, \
    get_environment_for_extents_3d
from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts, sequence_of_dicts_to_dict_of_np_arrays, \
    sequence_of_dicts_to_dict_of_tensors, index_dict_of_batched_vectors_tf
from state_space_dynamics import model_utils


def predict(scenario, fwd_models, classifier_model, environment, start_states, actions, n_actions, n_actions_sampled, n_start_states):
    """
    :param fwd_models:
    :param classifier_model:
    :param environment:
    :param start_states: a list of dicts, where each dict is a state
    :param actions: a list of lists of dicts. the first dimension should match the length of start_states
    :return:
    """
    actions_batched = {k: tf.reshape(v, [n_actions_sampled * n_start_states, n_actions, -1]) for k, v in actions.items()}
    start_states = make_dict_tf_float32({k: tf.expand_dims(v, axis=1) for k, v in start_states.items()})
    # copy from start states for each random action
    start_states_tiled = {k: tf.concat([v] * n_actions_sampled, axis=0) for k, v in start_states.items()}
    predictions = fwd_models.propagate_differentiable_batched(start_states=start_states_tiled, actions=actions_batched)
    environment_batched = {k: tf.stack([v] * n_actions_sampled, axis=0) for k, v in environment.items()}
    if classifier_model is not None:
        accept_probabilities = classifier_model.check_constraint_differentiable_batched_tf(environment=environment_batched,
                                                                                           predictions=predictions,
                                                                                           actions=actions)
    else:
        accept_probabilities = None
    # break out the num actions and num start states
    predictions = {k: tf.reshape(v, [n_start_states, n_actions_sampled, n_actions + 1, -1]) for k, v in predictions.items()}
    return predictions, accept_probabilities


def execute(service_provider, scenario, start_states, random_actions, n_start_states):
    actual_state_sequences = []
    for idx in range(n_start_states):
        actions = index_dict_of_batched_vectors_tf(random_actions, idx, batch_axis=0)
        start_state = index_dict_of_batched_vectors_tf(start_states, idx, batch_axis=0)
        # set_rope_state(service_provider, start_state)
        # service_provider.stop_robot(EmptyRequest())
        actual_state_sequence = execute_actions(service_provider, scenario, actions)
        actual_states_dict = sequence_of_dicts_to_dict_of_np_arrays(actual_state_sequence)
        actual_state_sequences.append(actual_states_dict)
    return actual_state_sequences


def set_rope_config(service_provider, start_config):
    if start_config is None:
        return
    x = start_config['x']
    y = start_config['y']
    yaw = np.deg2rad(start_config['yaw'])
    joint_angles = np.deg2rad(start_config['joint_angles'])
    for i in range(3):
        service_provider.reset_rope(x=x,
                                    y=y,
                                    yaw=yaw,
                                    joint_angles=joint_angles)


def setup(service_provider, fwd_model, test_config, real_time_rate: float = 0):
    max_step_size = fwd_model.dynamics_data_params.max_step_size
    service_provider.setup_env(verbose=0,
                               real_time_rate=real_time_rate,
                               max_step_size=max_step_size)

    movable_object_services = {k: make_movable_object_services(k) for k in test_config['object_positions'].keys()}
    fwd_model.scenario.move_objects_to_positions(movable_object_services, test_config['object_positions'])

    environment = get_environment_for_extents_3d(extent=fwd_model.full_env_params['extent'],
                                                 res=fwd_model.full_env_params['res'],
                                                 service_provider=service_provider,
                                                 robot_name=fwd_model.scenario.robot_name())
    return environment


def get_start_states(service_provider, fwd_model, desired_start_states):
    start_states = []
    for desired_start_state in desired_start_states:
        # in general we would need a planner to do this...
        # fwd_model.scenario.set_state(service_provider, desired_start_state)
        state = get_states_dict(service_provider, fwd_model.states_keys)
        state['stdev'] = np.array([0.0], dtype=np.float32)
        start_states.append(state)
    start_states = sequence_of_dicts_to_dict_of_tensors(start_states)
    return start_states


def predict_and_execute(service_provider, classifier_model, fwd_model, environment, start_states, actions, n_actions,
                        n_actions_sampled, n_start_states):
    # Prediction
    predicted_states, accept_probabilities = predict(fwd_model, classifier_model, environment, start_states, actions, n_actions,
                                                     n_actions_sampled, n_start_states)
    # Execute
    actual_states_list = execute(service_provider, fwd_model.scenario, start_states, actions, n_start_states)
    predicted_states_list = dict_of_sequences_to_sequence_of_dicts(predicted_states)
    return fwd_model, classifier_model, environment, actual_states_list, predicted_states_list, accept_probabilities


def load_models(classifier_model_dir, fwd_model_dir):
    fwd_model, _ = model_utils.load_generic_model(fwd_model_dir)
    if classifier_model_dir is not None:
        classifier_model = classifier_utils.load_generic_model(classifier_model_dir, fwd_model.scenario)
    else:
        classifier_model = None
    return classifier_model, fwd_model
