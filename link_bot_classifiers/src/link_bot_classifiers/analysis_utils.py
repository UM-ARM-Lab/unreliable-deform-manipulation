import numpy as np
import tensorflow as tf

from link_bot_classifiers import classifier_utils
from link_bot_gazebo.gazebo_services import GazeboServices
from link_bot_planning.plan_and_execute import execute_plan
from link_bot_pycommon.link_bot_sdf_utils import env_from_occupancy_data
from link_bot_pycommon.ros_pycommon import get_states_dict, get_occupancy_data
from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts, sequence_of_dicts_to_dict_of_np_arrays
from state_space_dynamics import model_utils
from std_srvs.srv import EmptyRequest


def predict(fwd_models, classifier_model, environment, state_dict, actions):
    n_actions_sampled = actions.shape[0]
    state_vec = state_dict['link_bot']
    state_batched = tf.expand_dims(tf.stack([state_vec] * n_actions_sampled, axis=0), axis=1)
    state_dict_batched = {
        'link_bot': state_batched,
    }
    predictions = fwd_models.propagate_differentiable_batched(start_states=state_dict_batched, actions=actions)
    environment_batched = {k: tf.stack([v] * n_actions_sampled, axis=0) for k, v in environment.items()}
    accept_probabilities = classifier_model.check_constraint_differentiable_batched_tf(environment=environment_batched,
                                                                                       predictions=predictions,
                                                                                       actions=actions)
    return predictions, accept_probabilities


def execute(service_provider, dt, start_configs, random_actions):
    actual_state_sequences = []
    for start_config, actions in zip(start_configs, random_actions):
        set_rope_config(start_config, service_provider)
        service_provider.stop_robot(EmptyRequest())
        actual_state_sequence = execute_plan(service_provider, dt, actions)
        actual_states_dict = sequence_of_dicts_to_dict_of_np_arrays(actual_state_sequence)
        actual_state_sequences.append(actual_states_dict)
    return actual_state_sequences


def set_rope_config(start_config, service_provider):
    if start_config is None:
        return
    x, y, yaw, joint_angles = start_config
    for i in range(3):
        service_provider.reset_rope(x=x,
                                    y=y,
                                    yaw=yaw,
                                    joint_angles=joint_angles)


def setup(service_provider, classifier_model, fwd_model, test_config, real_time_rate: float = 0):
    max_step_size = fwd_model.hparams['dynamics_dataset_hparams']['max_step_size']
    service_provider.setup_env(verbose=0,
                               real_time_rate=real_time_rate,
                               reset_robot=test_config['reset_robot'],
                               max_step_size=max_step_size,
                               stop=True,
                               reset_world=test_config['reset_world'])
    service_provider.move_objects_to_positions(test_config['object_positions'])
    full_env_data = get_occupancy_data(env_w_m=fwd_model.full_env_params.w,
                                       env_h_m=fwd_model.full_env_params.h,
                                       res=fwd_model.full_env_params.res,
                                       service_provider=service_provider,
                                       robot_name=fwd_model.scenario.robot_name())
    state = get_states_dict(service_provider, fwd_model.states_keys)
    if classifier_model.model_hparams['stdev']:
        state['stdev'] = np.array([0.0], dtype=np.float32)
    environment = env_from_occupancy_data(full_env_data)
    return environment, state


def predict_and_execute(classifier_model_dir, fwd_model_dir, test_config, start_configs, actions):
    """
    :return: the actual and predicted states lists are List of Dict where elements are [T,N]
    """
    fwd_model, _ = model_utils.load_generic_model(fwd_model_dir)
    classifier_model = classifier_utils.load_generic_model(classifier_model_dir, fwd_model.scenario)
    service_provider = GazeboServices(test_config['object_positions'].keys())
    environment, state = setup(service_provider, classifier_model, fwd_model, test_config)
    # Prediction
    predicted_states, accept_probabilities = predict(fwd_model, classifier_model, environment, state, actions)
    # Execute
    actual_states_list = execute(service_provider, fwd_model.dt, start_configs, actions)
    predicted_states_list = dict_of_sequences_to_sequence_of_dicts(predicted_states)
    return fwd_model, classifier_model, environment, actual_states_list, predicted_states_list, accept_probabilities
