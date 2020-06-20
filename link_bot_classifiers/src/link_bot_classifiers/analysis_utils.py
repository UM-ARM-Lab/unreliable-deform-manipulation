import tensorflow as tf

from link_bot_classifiers import classifier_utils
from link_bot_planning.plan_and_execute import execute_actions
from link_bot_pycommon.pycommon import make_dict_tf_float32
from link_bot_pycommon.ros_pycommon import make_movable_object_services, \
    get_environment_for_extents_3d
from moonshine.moonshine_utils import dict_of_sequences_to_sequence_of_dicts, sequence_of_dicts_to_dict_of_np_arrays, \
    index_dict_of_batched_vectors_tf, numpify
from state_space_dynamics import model_utils
from state_space_dynamics.model_utils import EnsembleDynamicsFunction


def predict(fwd_models, environment, start_states, actions, n_actions, n_actions_sampled, n_start_states):
    actions_batched = {k: tf.reshape(v, [n_actions_sampled * n_start_states, n_actions, -1]) for k, v in actions.items()}
    start_states = make_dict_tf_float32({k: tf.expand_dims(v, axis=1) for k, v in start_states.items()})
    # copy from start states for each random action
    start_states_tiled = {k: tf.concat([v] * n_actions_sampled, axis=0) for k, v in start_states.items()}
    predictions = fwd_models.propagate_differentiable_batched(start_states=start_states_tiled, actions=actions_batched)
    # break out the num actions and num start states
    predictions = {k: tf.reshape(v, [n_start_states, n_actions_sampled, n_actions + 1, -1]) for k, v in predictions.items()}
    return predictions


def predict_classifier(classifier_model, environment, predictions, actions, n_actions_sampled):
    environment_batched = {k: tf.stack([v] * n_actions_sampled, axis=0) for k, v in environment.items()}
    accept_probabilities = classifier_model.check_constraint_differentiable_batched_tf(environment=environment_batched,
                                                                                       predictions=predictions,
                                                                                       actions=actions)
    return accept_probabilities


def predict_and_classify(fwd_models, classifier_model, environment, start_states, actions, n_actions, n_actions_sampled,
                         n_start_states):
    predictions = predict(fwd_models, environment, start_states, actions, n_actions, n_actions_sampled, n_start_states)
    accept_probabilities = predict_classifier(classifier_model, environment, predictions, actions, n_actions_sampled)
    return predictions, accept_probabilities


def execute(service_provider, scenario, start_states, random_actions, n_actions_sampled, n_start_states):
    actual_state_sequences = []
    for i in range(n_start_states):
        actual_state_sequences_for_start_state = []
        actions_for_start_state = index_dict_of_batched_vectors_tf(random_actions, i, batch_axis=0)
        start_state = index_dict_of_batched_vectors_tf(start_states, i, batch_axis=0)
        for j in range(n_actions_sampled):
            actions = index_dict_of_batched_vectors_tf(actions_for_start_state, j, batch_axis=0)
            # reset to the start state
            scenario.teleport_to_state(numpify(start_state))
            # execute actions and record the observed states
            actual_state_sequence = execute_actions(service_provider, scenario, actions)
            actual_states_dict = sequence_of_dicts_to_dict_of_np_arrays(actual_state_sequence)
            actual_state_sequences_for_start_state.append(actual_states_dict)
        # reset when done for conveniently re-running the script
        actual_state_sequences.append(actual_state_sequences_for_start_state)
        scenario.teleport_to_state(numpify(start_state))
    return actual_state_sequences


def setup(service_provider, fwd_model: EnsembleDynamicsFunction, test_config, real_time_rate: float = 0):
    max_step_size = fwd_model.data_collection_params['max_step_size']
    service_provider.setup_env(verbose=0, real_time_rate=real_time_rate, max_step_size=max_step_size)

    movable_object_services = {k: make_movable_object_services(k) for k in test_config['object_positions'].keys()}
    fwd_model.scenario.move_objects_to_positions(movable_object_services, test_config['object_positions'])

    environment = get_environment_for_extents_3d(extent=fwd_model.data_collection_params['extent'],
                                                 res=fwd_model.data_collection_params['res'],
                                                 service_provider=service_provider,
                                                 robot_name=fwd_model.scenario.robot_name())
    return environment


def predict_and_execute(service_provider, classifier_model, fwd_model: EnsembleDynamicsFunction, environment, start_states,
                        actions, n_actions, n_actions_sampled, n_start_states):
    # Prediction
    predicted_states, accept_probabilities = predict(fwd_models=fwd_model,
                                                     classifier_model=classifier_model,
                                                     environment=environment,
                                                     start_states=start_states,
                                                     actions=actions,
                                                     n_actions=n_actions,
                                                     n_actions_sampled=n_actions_sampled,
                                                     n_start_states=n_start_states)
    # Execute
    scenario = fwd_model.scenario
    actual_states_lists = execute(service_provider, scenario, start_states, actions, n_actions_sampled, n_start_states)
    predicted_states_list = dict_of_sequences_to_sequence_of_dicts(predicted_states)
    predicted_states_lists = [dict_of_sequences_to_sequence_of_dicts(p) for p in predicted_states_list]

    return fwd_model, classifier_model, environment, actual_states_lists, predicted_states_lists, accept_probabilities


def load_models(classifier_model_dir, fwd_model_dir):
    fwd_model, _ = model_utils.load_generic_model(fwd_model_dir)
    if classifier_model_dir is not None:
        classifier_model = classifier_utils.load_generic_model(classifier_model_dir, fwd_model.scenario)
    else:
        classifier_model = None
    return classifier_model, fwd_model
