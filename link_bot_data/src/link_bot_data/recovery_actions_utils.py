from typing import Dict

import numpy as np
import rospy
from colorama import Fore
import tensorflow as tf
from time import perf_counter
from link_bot_classifiers.nn_classifier import NNClassifierWrapper
from link_bot_data.classifier_dataset_utils import \
    batch_of_many_of_actions_sequences_to_dict
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_pycommon import rviz_animation_controller
from link_bot_pycommon.pycommon import make_dict_tf_float32
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import (add_batch, gather_dict,
                                       index_dict_of_batched_vectors_tf,
                                       remove_batch,
                                       sequence_of_dicts_to_dict_of_tensors)
from state_space_dynamics.model_utils import EnsembleDynamicsFunction
from std_msgs.msg import Float32


def generate_recovery_examples(fwd_model: EnsembleDynamicsFunction,
                               classifier_model: NNClassifierWrapper,
                               tf_dataset: tf.data.Dataset,
                               dataset: DynamicsDataset,
                               labeling_params: Dict,
                               batch_size: int,
                               start_at: int,
                               stop_at: int):
    action_sequence_horizon = labeling_params['action_sequence_horizon']
    tf_dataset = tf_dataset.batch(batch_size)
    action_rng = np.random.RandomState(0)
    n_batches = 0
    for _ in tf_dataset:
        n_batches += 1

    t0 = perf_counter()
    for in_batch_idx, example in enumerate(tf_dataset):
        if start_at is not None and in_batch_idx < start_at:
            continue
        if stop_at is not None and in_batch_idx >= stop_at:
            print(Fore.GREEN + "Done!" + Fore.RESET)
            raise StopIteration()
        dt = perf_counter() - t0
        print(Fore.GREEN + f"{in_batch_idx}/{n_batches}, {dt:.3f}s" + Fore.RESET)
        actual_batch_size = int(example['traj_idx'].shape[0])
        # iterate over every subsequence of exactly length actions_sequence_horizon
        for start_t in range(0, dataset.sequence_length - action_sequence_horizon + 1, labeling_params['start_step']):
            end_t = start_t + action_sequence_horizon

            actual_states_from_start_t = {k: example[k][:, start_t:end_t] for k in fwd_model.state_keys}
            actions_from_start_t = {k: example[k][:, start_t:end_t - 1] for k in fwd_model.action_keys}

            data = (example,
                    actions_from_start_t,
                    actual_states_from_start_t,
                    labeling_params,
                    dataset.data_collection_params,
                    )
            constants = (actual_batch_size,
                         action_sequence_horizon,
                         classifier_model.horizon,
                         actual_batch_size,
                         start_t,
                         end_t)
            out_examples = generate_recovery_actions_examples(fwd_model, classifier_model, data, constants, action_rng)
            yield out_examples


def generate_recovery_actions_examples(fwd_model, classifier_model, data, constants, action_rng):
    example, actual_actions, actual_states, labeling_params, data_collection_params = data
    actual_batch_size, action_sequence_horizon, classifier_horizon, batch_size, start_t, end_t = constants
    scenario = fwd_model.scenario

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

    classifier_rejects = []
    all_accept_probabilities = []
    for t in range(action_sequence_horizon):
        # Sample actions
        n_action_samples = labeling_params['n_action_samples']
        n_actions = classifier_horizon - 1
        actual_states_t = index_dict_of_batched_vectors_tf(actual_states, t, batch_axis=1)
        random_actions_dict = scenario.batch_stateless_sample_action(environment=environment,
                                                                     state=actual_states_t,
                                                                     batch_size=actual_batch_size,
                                                                     n_action_samples=n_action_samples,
                                                                     n_actions=n_actions,
                                                                     data_collection_params=data_collection_params,
                                                                     action_params=data_collection_params,
                                                                     action_rng=action_rng)
        batch_sample = actual_batch_size * n_action_samples
        random_actions_dict = {k: tf.reshape(v, [batch_sample, n_actions, -1])
                               for k, v in random_actions_dict.items()}

        def _predict_and_classify(_actual_states, _random_actions_dict):
            # [t:t+1] to keep dim, as opposed to just [t]
            start_states_tiled = {k: tf.tile(v[:, t:t+1, :], [n_action_samples, 1, 1])
                                  for k, v in _actual_states.items()}

            # Predict
            predictions = fwd_model.propagate_differentiable_batched(
                start_states=start_states_tiled, actions=_random_actions_dict)

            # Check classifier
            environment_tiled = {k: tf.concat([v] * n_action_samples, axis=0) for k, v in environment.items()}
            accept_probabilities = classifier_model.check_constraint_batched_tf(environment=environment_tiled,
                                                                                predictions=predictions,
                                                                                actions=_random_actions_dict,
                                                                                batch_size=batch_sample,
                                                                                state_sequence_length=classifier_horizon)
            return predictions, accept_probabilities

        predictions, accept_probabilities = _predict_and_classify(actual_states, random_actions_dict)

        # # BEGIN DEBUG
        # accept_prob_pub_ = rospy.Publisher("accept_probability_viz", Float32, queue_size=10)
        # for b in range(actual_batch_size):
        #     environment_b = index_dict_of_batched_vectors_tf(environment, b)
        #     actual_state_b = index_dict_of_batched_vectors_tf(actual_states, b)
        #     actual_state_b_t = index_dict_of_batched_vectors_tf(actual_state_b, t)
        #     for s in range(n_action_samples):
        #         time_steps = np.arange(classifier_horizon)
        #         scenario.plot_environment_rviz(environment_b)
        #         anim = RvizAnimationController(time_steps)
        #         ravel_batch_idx = np.ravel_multi_index(dims=[actual_batch_size, n_action_samples], multi_index=[b, s])

        #         scenario.plot_state_rviz(actual_state_b_t, label='start', color='#ffff00aa')
        #         while not anim.done:
        #             h = anim.t()
        #             pred_b_a_s = index_dict_of_batched_vectors_tf(predictions, ravel_batch_idx)
        #             action_b_a_s = index_dict_of_batched_vectors_tf(random_actions_dict, ravel_batch_idx)
        #             pred_t = remove_batch(scenario.index_state_time(add_batch(pred_b_a_s), h))

        #             if h > 0:
        #                 accept_prob_t = accept_probabilities[ravel_batch_idx, h - 1].numpy()
        #             else:
        #                 accept_prob_t = -999
        #             accept_prob_msg = Float32()
        #             accept_prob_msg.data = accept_prob_t
        #             accept_prob_pub_.publish(accept_prob_msg)

        #             color = "#ff0000aa" if accept_prob_t < 0.5 else "#00ff00aa"
        #             scenario.plot_state_rviz(pred_t, label='predicted', color=color)
        #             if h < anim.max_t:
        #                 action_t = remove_batch(scenario.index_action_time(add_batch(action_b_a_s), h))
        #                 scenario.plot_action_rviz(pred_t, action_t)
        #             else:
        #                 action_t = remove_batch(scenario.index_action_time(add_batch(action_b_a_s), h - 1))
        #                 prev_pred_t = remove_batch(scenario.index_state_time(add_batch(pred_b_a_s), h - 1))
        #                 scenario.plot_action_rviz(prev_pred_t, action_t)

        #             anim.step()
        # # END DEBUG

        # reshape to separate batch from sampled actions
        accept_probabilities = tf.reshape(accept_probabilities, [batch_size, n_action_samples])

        # a time step needs recovery if every time step of every sampled random action sequence was rejected by the classifier
        # needs_recovery has shape [batch size, action_sequence_horizon]
        classifier_rejects_t = accept_probabilities < 0.5
        all_accept_probabilities.append(accept_probabilities)
        classifier_rejects.append(classifier_rejects_t)

    # an example is recovering if at the first time step (axis 1) needs_recovery is true, and if at some point later in time
    # needs_recovery is false
    classifier_rejects = tf.stack(classifier_rejects, axis=1)
    all_accept_probabilities = tf.stack(all_accept_probabilities, axis=1)
    first_time_step_needs_recovery = tf.reduce_all(classifier_rejects[:, 0], axis=-1)
    valid_example = first_time_step_needs_recovery

    # construct output examples dict
    out_examples = {
        'env': full_env,
        'origin': full_env_origin,
        'extent': full_env_extent,
        'res': full_env_res,
        'traj_idx': example['traj_idx'],
        'start_t': tf.stack([start_t] * batch_size),
        'end_t': tf.stack([end_t] * batch_size),
        'accept_probabilities': all_accept_probabilities
    }
    # add true start states
    out_examples.update(actual_states)
    out_examples.update(actual_actions)
    out_examples = make_dict_tf_float32(out_examples)

    valid_indices = tf.squeeze(tf.where(valid_example), axis=1)
    valid_out_examples = gather_dict(out_examples, valid_indices)

    for b in range(tf.size(valid_indices)):
        score = tf.math.count_nonzero(all_accept_probabilities[b][1] > 0.5) / n_action_samples
        print(f"score {score.numpy()}")

    # # BEGIN DEBUG
    # for b in range(tf.size(valid_indices)):
    #     valid_out_example_b = index_dict_of_batched_vectors_tf(valid_out_examples, b)
    #     scenario.plot_environment_rviz(valid_out_example_b)
    #     score = tf.math.count_nonzero(all_accept_probabilities[b][1] > 0.5) / n_action_samples

    #     anim = RvizAnimationController(np.arange(action_sequence_horizon))
    #     while not anim.done:
    #         t = anim.t()
    #         s_t = {k: valid_out_example_b[k][t] for k in actual_states.keys()}
    #         scenario.plot_state_rviz(s_t, label='start', color='#ff0000')
    #         if t < anim.max_t:
    #             a_t = {k: valid_out_example_b[k][t] for k in actual_actions.keys()}
    #             scenario.plot_action_rviz(s_t, a_t)
    #         anim.step()
    # # END DEBUG

    return valid_out_examples


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
