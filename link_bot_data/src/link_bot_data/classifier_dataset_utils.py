from typing import Dict

import numpy as np
import tensorflow as tf

from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import null_pad, add_planned, null_diverged, null_previous_states
from moonshine.moonshine_utils import index_dict_of_batched_vectors_tf


def add_model_predictions(fwd_model,
                          tf_dataset: tf.data.TFRecordDataset,
                          dataset: DynamicsDataset,
                          labeling_params: Dict):
    prediction_horizon = labeling_params['prediction_horizon']
    classifier_horizon = labeling_params['classifier_horizon']
    assert classifier_horizon >= 2
    assert prediction_horizon <= dataset.desired_sequence_length
    batch_size = 1024
    for dataset_element in tf_dataset.batch(batch_size):
        inputs, outputs = dataset_element
        actual_batch_size = int(inputs['traj_idx'].shape[0])

        for prediction_start_t in range(0, dataset.max_sequence_length - classifier_horizon - 1, labeling_params['start_step']):
            prediction_end_t = min(prediction_start_t + prediction_horizon, dataset.max_sequence_length)
            actual_prediction_horizon = prediction_end_t - prediction_start_t
            outputs_from_start_t = {k: v[:, prediction_start_t:prediction_end_t] for k, v in outputs.items()}

            predictions_from_start_t = predict_subsequence(states_description=dataset.states_description,
                                                           fwd_model=fwd_model,
                                                           dataset_element=dataset_element,
                                                           prediction_start_t=prediction_start_t,
                                                           prediction_horizon=prediction_horizon)

            for batch_idx in range(actual_batch_size):
                inputs_b = index_dict_of_batched_vectors_tf(inputs, batch_idx)
                outputs_from_start_t_b = index_dict_of_batched_vectors_tf(outputs_from_start_t, batch_idx)
                predictions_from_start_t_b = index_dict_of_batched_vectors_tf(predictions_from_start_t, batch_idx)
                yield from generate_examples_for_prediction(inputs=inputs_b,
                                                            outputs=outputs_from_start_t_b,
                                                            predictions=predictions_from_start_t_b,
                                                            start_t=prediction_start_t,
                                                            labeling_params=labeling_params,
                                                            prediction_horizon=actual_prediction_horizon)


def generate_examples_for_prediction(inputs: Dict,
                                     outputs: Dict,
                                     predictions: Dict,
                                     start_t: int,
                                     labeling_params: Dict,
                                     prediction_horizon: int):
    classifier_horizon = labeling_params['classifier_horizon']
    for classifier_start_t in range(0, prediction_horizon - classifier_horizon + 1):
        classifier_end_t = min(classifier_start_t + classifier_horizon, prediction_horizon)

        full_env = inputs['full_env/env']
        full_env_origin = inputs['full_env/origin']
        full_env_extent = inputs['full_env/extent']
        full_env_res = inputs['full_env/res']
        traj_idx = inputs['traj_idx']
        out_example = {
            'full_env/env': full_env,
            'full_env/origin': full_env_origin,
            'full_env/extent': full_env_extent,
            'full_env/res': full_env_res,
            'traj_idx': tf.squeeze(traj_idx),
            'prediction_start_t': start_t,
            'classifier_start_t': classifier_start_t,
            'classifier_end_t': classifier_end_t,
        }

        # this slice gives arrays of fixed length (ex, 5) which must be null padded from out_example_end_idx onwards
        state_slice = slice(classifier_start_t, classifier_start_t + classifier_horizon)
        action_slice = slice(classifier_start_t, classifier_start_t + classifier_horizon - 1)
        sliced_outputs = {}
        for name, output in outputs.items():
            output_from_cst = output[state_slice]
            out_example[name] = output_from_cst
            sliced_outputs[name] = output_from_cst

        sliced_predictions = {}
        for name, prediction in predictions.items():
            pred_from_cst = prediction[state_slice]
            out_example[add_planned(name)] = pred_from_cst
            sliced_predictions[name] = pred_from_cst

        # action
        if 'action' in inputs:
            actions = inputs['action'][action_slice]
            out_example['action'] = actions

        # TODO: planned -> predicted whenever we're talking about applying the dynamics model
        #  so like add_planned should be add_predicted or something

        # compute label
        is_close = compute_is_close_tf(actual_states_dict=sliced_outputs,
                                       predicted_states_dict=sliced_predictions,
                                       labeling_params=labeling_params)
        is_first_valid_state_close = is_close[0]
        out_example['is_close'] = tf.cast(is_close, dtype=tf.float32)

        if is_first_valid_state_close:
            yield out_example
        elif classifier_horizon > 5:
            # it's really unlikely things will reconverge here so consider return saves uselessly iterating over a ton of data
            return


def compute_is_close_tf(actual_states_dict: Dict, labeling_params: Dict, predicted_states_dict: Dict):
    state_key = labeling_params['state_key']
    labeling_states = tf.convert_to_tensor(actual_states_dict[state_key])
    labeling_predicted_states = tf.convert_to_tensor(predicted_states_dict[state_key])
    # TODO: use scenario to compute distance here?
    model_error = tf.linalg.norm(labeling_states - labeling_predicted_states, axis=1)
    threshold = labeling_params['threshold']
    is_close = model_error < threshold
    return is_close


def compute_label_np(actual_states_dict: Dict, labeling_params: Dict, predicted_states_dict: Dict):
    is_close = compute_is_close_tf(actual_states_dict, labeling_params, predicted_states_dict)
    return is_close.numpy()


def predict_subsequence(states_description, fwd_model, dataset_element, prediction_start_t, prediction_horizon):
    inputs, outputs = dataset_element

    # build inputs to the network
    actions = inputs['action'][:, prediction_start_t:prediction_start_t + prediction_horizon - 1]  # one fewer actions than states
    start_states_t = {}
    for name in states_description.keys():
        start_state_t = tf.expand_dims(inputs[name][:, prediction_start_t], axis=1)
        start_states_t[name] = start_state_t

    # call the network
    predictions = fwd_model.propagate_differentiable_batched(start_states_t, actions)
    return predictions


def predict_and_nullify(dataset, fwd_model, dataset_element, labeling_params, batch_size, start_t):
    inputs, outputs = dataset_element
    actions = inputs['action']
    start_states_t = {}
    for name in dataset.states_description.keys():
        start_state_t = tf.expand_dims(inputs[name][:, start_t], axis=1)
        start_states_t[name] = start_state_t
    predictions_from_start_t = fwd_model.propagate_differentiable_batched(start_states_t, actions[:, start_t:])
    if labeling_params['discard_diverged']:
        # null out all the predictions past divergence
        predictions_from_start_t, last_valid_ts = null_diverged(outputs,
                                                                predictions_from_start_t,
                                                                start_t,
                                                                labeling_params)
    else:
        # an array of size batch equal to the time-sequence length of outputs
        last_valid_ts = np.ones(batch_size) * (dataset.desired_sequence_length - 1)
    # when start_t > 0, this output will need to be padded so that all outputs are the same size
    all_predictions = null_previous_states(predictions_from_start_t, dataset.desired_sequence_length)
    return all_predictions, last_valid_ts
