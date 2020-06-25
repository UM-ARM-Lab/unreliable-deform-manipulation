from typing import Dict

import tensorflow as tf

from link_bot_classifiers.nn_classifier import NNClassifierWrapper
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_pycommon.pycommon import make_dict_tf_float32
from moonshine.moonshine_utils import gather_dict
from state_space_dynamics.model_utils import EnsembleDynamicsFunction


def generate_recovery_examples(fwd_model: EnsembleDynamicsFunction,
                               classifier_model: NNClassifierWrapper,
                               tf_dataset: tf.data.Dataset,
                               dataset: DynamicsDataset,
                               labeling_params: Dict):
    batch_size = 256
    action_sequence_horizon = labeling_params['action_sequence_horizon']
    for example in tf_dataset.batch(batch_size):
        actual_batch_size = int(example['traj_idx'].shape[0])
        # iterate over every subsequence of exactly length actions_sequence_horizon
        for start_t in range(0, dataset.sequence_length - action_sequence_horizon - 1, labeling_params['start_step']):
            end_t = start_t + action_sequence_horizon
            print(start_t, end_t)

            actual_states_from_start_t = {k: example[k][:, start_t:end_t] for k in fwd_model.state_keys}
            actions_from_start_t = {k: example[k][:, start_t:end_t - 1] for k in fwd_model.action_keys}

            data = (example,
                    actions_from_start_t,
                    actual_states_from_start_t,
                    labeling_params,
                    )
            constants = (actual_batch_size,
                         action_sequence_horizon,
                         classifier_model.horizon,
                         actual_batch_size,
                         start_t,
                         end_t)
            out_examples = generate_recovery_actions_examples(fwd_model, classifier_model, data, constants)
            if out_examples is not None:
                yield out_examples


def generate_recovery_actions_examples(fwd_model, classifier_model, data, constants):
    example, actual_actions, actual_states, labeling_params = data
    actual_batch_size, action_sequence_horizon, classifier_horizon, batch_size, start_t, end_t = constants
    print('classifier horizon =', classifier_horizon)

    full_env = example['env']
    full_env_origin = example['origin']
    full_env_extent = example['extent']
    full_env_res = example['res']
    environment = {
        'env': full_env,
        'origin': full_env_origin,
        'extent': full_env_extent,
        'res': full_env_res,
    }

    # Sample actions
    n_action_samples = labeling_params['n_action_samples']
    n_actions = classifier_horizon - 1
    samples_and_time = n_action_samples * action_sequence_horizon
    # TODO: because sampling an action is state dependant, a more "correct" way to sample a sequence of actions is to:
    #   sample 1 action, propagate, and repeat.
    random_actions = []
    for i in range(action_sequence_horizon):
        actual_state = fwd_model.scenario.index_state_time(actual_states, i)
        random_actions_for_start_state = fwd_model.scenario.sample_actions(environment=environment,
                                                                           start_state=actual_state,
                                                                           action_sequence_length=n_actions,
                                                                           n_action_samples=n_action_samples)
        random_actions.append(random_actions_for_start_state)
    start_states_tiled = {k: tf.tile(v[:, 0:1, :], [samples_and_time, 1, 1]) for k, v in actual_states.items()}  # 0:1 to keep dim

    # Predict
    predictions = fwd_model.propagate_differentiable_batched(start_states=start_states_tiled, actions=random_actions)

    # Check classifier
    environment_tiled = {k: tf.concat([v] * samples_and_time, axis=0) for k, v in environment.items()}
    accept_probabilities = classifier_model.check_constraint_batched_tf(environment=environment_tiled,
                                                                        predictions=predictions,
                                                                        actions=random_actions)
    # reshape to separate batch from sampled actions
    accept_probabilities = tf.reshape(accept_probabilities,
                                      [batch_size, action_sequence_horizon, n_action_samples, classifier_horizon - 1])

    # a time step needs recovery if every time step of every sampled random action sequence was rejected by the classifier
    # needs_recovery has shape [batch size, action_sequence_horizon]
    needs_recovery = tf.reduce_all(tf.reduce_all(accept_probabilities < 0.5, axis=2), axis=2)

    # an example is recovering if at the first time step (axis 1) needs_recovery is true, and if at some point later in time
    # needs_recovery is false
    first_time_step_needs_recovery = needs_recovery[:, 0]
    later_time_step_doesnt_need_recovery = tf.logical_not(tf.reduce_any(needs_recovery[:, 1:], axis=1))
    valid_example = tf.logical_and(first_time_step_needs_recovery, later_time_step_doesnt_need_recovery)
    needs_recovery_int = tf.cast(needs_recovery, tf.int32)
    mask = recovering_mask(needs_recovery_int)

    # construct output examples dict
    out_examples = {
        'env': full_env,
        'origin': full_env_origin,
        'extent': full_env_extent,
        'res': full_env_res,
        'traj_idx': example['traj_idx'][:, 0],
        'start_t': tf.stack([start_t] * batch_size),
        'end_t': tf.stack([end_t] * batch_size),
        'action': actual_actions[:, :, :-1],  # skip the last action
        'mask': mask,
    }
    # add true start states
    out_examples.update(actual_states)
    out_examples = make_dict_tf_float32(out_examples)

    valid_indices = tf.where(valid_example)
    if tf.greater(tf.size(valid_indices), 0):
        valid_out_examples = gather_dict(out_examples, valid_indices)
        return valid_out_examples
    else:
        return None


def recovering_mask(needs_recovery):
    """
    Looks for the first occurrence of the pattern [1, 0] in each row (appending a 0 to each row first),
    and the index where this is found defines the end of the mask.
    The first time step is masked to False because it will always be 1
    :param needs_recovery: float matrix [B,H] but all values should be 0.0 or 1.0
    :return: boolean matrix [B,H]
    """
    batch_size, horizon = needs_recovery.shape

    range = tf.stack([tf.range(horizon)] * batch_size, axis=0)
    mask = tf.cumsum(needs_recovery, axis=1) > range
    has_a_0 = tf.reduce_any(needs_recovery == 0, axis=1, keepdims=True)
    starts_with_1 = needs_recovery[:, 0:1] == 1
    trues = tf.cast(tf.ones([batch_size, 1]), tf.bool)
    mask_final = tf.concat([trues, mask[:, :-1]], axis=1)
    mask_final = tf.logical_and(tf.logical_and(mask_final, has_a_0), starts_with_1)
    return mask_final
