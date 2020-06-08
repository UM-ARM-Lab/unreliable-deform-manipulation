from typing import Dict

import tensorflow as tf

from link_bot_data.classifier_dataset_utils import predictions_vs_actual_generator, PredictionActualExample, compute_is_close_tf
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import add_planned
from link_bot_pycommon.pycommon import print_dict
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import gather_dict

limit_gpu_mem(2)


def generate_recovery_examples(fwd_model,
                               tf_dataset: tf.data.TFRecordDataset,
                               dataset: DynamicsDataset,
                               labeling_params: Dict):
    batch_size = 1024
    for prediction_actual in predictions_vs_actual_generator(fwd_model=fwd_model,
                                                             tf_dataset=tf_dataset,
                                                             batch_size=batch_size,
                                                             dataset=dataset,
                                                             prediction_function=predict_onestep,
                                                             labeling_params=labeling_params):
        yield from generate_recovery_actions_examples(prediction_actual)


def starts_recovering(is_close):
    return tf.argmax(is_close[1:]) > 0


def is_recovering(is_close):
    """
    tests if a binary sequence looks like 0 ... 0 1 ... 1
    first dim is batch
    :arg is_close [B, H] matrix of floats, but they must be 0.0 or 1.0
    """
    index_of_first_1 = tf.math.argmax(is_close, axis=1)
    num_ones = tf.cast(tf.math.reduce_sum(is_close, axis=1), tf.int64)
    num_elements_after_first_1 = is_close.shape[1] - index_of_first_1
    at_least_one_close = tf.math.reduce_sum(is_close, axis=1) > 0
    at_least_one_far = tf.math.reduce_sum(is_close, axis=1) < is_close.shape[1]
    recovering = tf.equal(num_ones - num_elements_after_first_1, 0)
    recovering = tf.logical_and(recovering, at_least_one_close)
    recovering = tf.logical_and(recovering, at_least_one_far)
    return recovering


def recovering_mask(is_close):
    """
    Looks for the first occurrence of the pattern [1, 0] in each row (appending a 0 to each row first),
    and the index where this is found defines the end of the mask.
    The first time step is masked to False because it will always be 1
    :param is_close: float matrix [B,H] but all values should be 0.0 or 1.0
    :return: boolean matrix [B,H]
    """
    batch_size = is_close.shape[0]
    # trim the first element and append a zero
    zeros = tf.zeros([batch_size, 1])
    trimmed_and_padded = tf.concat([is_close[:, 1:], zeros], axis=1)
    filters = tf.constant([[[1]], [[-1]]], dtype=tf.float32)
    conv_out = tf.squeeze(tf.nn.conv1d(tf.expand_dims(trimmed_and_padded, axis=2), filters, stride=1, padding='VALID'), axis=2)
    # conv_out is > 0 if the pattern [1,0] is found, otherwise it will be 0 or -1
    matches = tf.cast(conv_out > 0, tf.float32)
    shifted = tf.concat([zeros, matches[:, :-1]], axis=1)
    mask = tf.logical_not(tf.cast(tf.clip_by_value(tf.cumsum(shifted, axis=1), 0, 1), tf.bool))
    has_a_1 = tf.reduce_any(is_close[:, 1:] > 0, axis=1, keepdims=True)
    mask_and_has_1 = tf.logical_and(mask, has_a_1)
    return tf.concat((tf.cast(zeros, tf.bool), mask_and_has_1), axis=1)


def generate_recovery_actions_examples(prediction_actual: PredictionActualExample):
    inputs = prediction_actual.inputs
    outputs = prediction_actual.outputs
    predictions = prediction_actual.predictions
    labeling_params = prediction_actual.labeling_params
    prediction_horizon = prediction_actual.actual_prediction_horizon
    classifier_horizon = labeling_params['classifier_horizon']
    for classifier_start_t in range(0, prediction_horizon - classifier_horizon + 1):
        classifier_end_t = min(classifier_start_t + classifier_horizon, prediction_horizon)

        full_env = inputs['full_env/env']
        full_env_origin = inputs['full_env/origin']
        full_env_extent = inputs['full_env/extent']
        full_env_res = inputs['full_env/res']
        traj_idx = inputs['traj_idx']
        prediction_start_t = prediction_actual.prediction_start_t
        prediction_start_t_batched = tf.cast(tf.stack([prediction_start_t] * prediction_actual.batch_size, axis=0), tf.float32)
        classifier_start_t_batched = tf.cast(tf.stack([classifier_start_t] * prediction_actual.batch_size, axis=0), tf.float32)
        classifier_end_t_batched = tf.cast(tf.stack([classifier_end_t] * prediction_actual.batch_size, axis=0), tf.float32)
        out_example = {
            'full_env/env': full_env,
            'full_env/origin': full_env_origin,
            'full_env/extent': full_env_extent,
            'full_env/res': full_env_res,
            'traj_idx': tf.squeeze(traj_idx),
            'prediction_start_t': prediction_start_t_batched,
            'classifier_start_t': classifier_start_t_batched,
            'classifier_end_t': classifier_end_t_batched,
        }

        # this slice gives arrays of fixed length (ex, 5)
        state_slice = slice(classifier_start_t, classifier_start_t + classifier_horizon)
        action_slice = slice(classifier_start_t, classifier_start_t + classifier_horizon - 1)
        sliced_outputs = {}
        for name, output in outputs.items():
            output_from_cst = output[:, state_slice]
            out_example[name] = output_from_cst
            sliced_outputs[name] = output_from_cst

        sliced_predictions = {}
        for name, prediction in predictions.items():
            pred_from_cst = prediction[:, state_slice]
            out_example[add_planned(name)] = pred_from_cst
            sliced_predictions[name] = pred_from_cst

        # action
        actions = prediction_actual.actions[:, action_slice]
        out_example['action'] = actions

        # compute label
        is_close = compute_is_close_tf(actual_states_dict=sliced_outputs,
                                       predicted_states_dict=sliced_predictions,
                                       labeling_params=labeling_params)
        is_close_float = tf.cast(is_close, dtype=tf.float32)
        out_example['is_close'] = is_close_float
        mask = recovering_mask(is_close_float)
        out_example['mask'] = tf.cast(mask, tf.float32)

        is_first_predicted_state_close = is_close[:, 1]
        valid_indices = tf.where(is_first_predicted_state_close)
        valid_indices = tf.squeeze(valid_indices, axis=1)
        # keep only valid_indices from every key in out_example...
        valid_out_example = gather_dict(out_example, valid_indices)

        yield valid_out_example


def predict_onestep(states_description, fwd_model, dataset_element, batch_size, prediction_start_t, prediction_horizon):
    inputs, outputs = dataset_element

    # build inputs to the network
    input_states_t = {}
    for name in states_description.keys():
        input_states_t[name] = []
    actions = []
    for t in range(prediction_start_t, prediction_start_t + prediction_horizon - 1):  # one fewer actions than states
        actions.append(tf.expand_dims(inputs['action'][:, t], axis=1))
        for name in states_description.keys():
            input_state_t = tf.expand_dims(inputs[name][:, t], axis=1)
            input_states_t[name].append(input_state_t)

    input_states_t = {k: tf.concat(v, axis=0) for k, v in input_states_t.items()}
    actions = tf.concat(actions, axis=0)

    # call the network
    predictions = fwd_model.propagate_differentiable_batched(input_states_t, actions)
    # reshape to seperate batch and time
    out_shape = [prediction_horizon - 1, batch_size, -1]
    predictions = {k: tf.transpose(tf.reshape(v[:, 1, :], out_shape), [1, 0, 2]) for k, v in predictions.items()}
    predictions_with_start = {}
    for name, prediction in predictions.items():
        if name == 'stdev':
            input_state_t = tf.zeros([batch_size, 1, 1])
        else:
            input_state_t = tf.expand_dims(inputs[name][:, prediction_start_t], axis=1)
        predictions_with_start[name] = tf.concat([input_state_t, prediction], axis=1)
    return predictions_with_start
