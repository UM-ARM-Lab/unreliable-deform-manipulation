from typing import Dict

import tensorflow as tf

from link_bot_data.classifier_dataset_utils import PredictionActualExample, compute_is_close_tf
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import add_planned
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import remove_batch, add_batch, index_dict_of_batched_vectors_tf

limit_gpu_mem(2)


def one_step_prediction_actual_generator(fwd_model,
                                         tf_dataset,
                                         batch_size: int,
                                         dataset: DynamicsDataset,
                                         labeling_params: Dict):
    prediction_horizon = labeling_params['prediction_horizon']
    classifier_horizon = labeling_params['classifier_horizon']
    assert classifier_horizon >= 2
    assert prediction_horizon <= dataset.desired_sequence_length
    for dataset_element in tf_dataset.batch(batch_size):
        inputs, outputs = dataset_element
        actual_batch_size = int(inputs['traj_idx'].shape[0])

        for prediction_start_t in range(0, dataset.max_sequence_length - classifier_horizon - 1, labeling_params['start_step']):
            prediction_end_t = min(prediction_start_t + prediction_horizon, dataset.max_sequence_length)
            actual_prediction_horizon = prediction_end_t - prediction_start_t
            outputs_from_start_t = {k: v[:, prediction_start_t:prediction_end_t] for k, v in outputs.items()}
            actions_from_start_t = inputs['action'][:, prediction_start_t:prediction_end_t]

            predictions_from_start_t = predict_onestep(states_description=dataset.states_description,
                                                       fwd_model=fwd_model,
                                                       dataset_element=dataset_element,
                                                       batch_size=batch_size,
                                                       prediction_start_t=prediction_start_t,
                                                       prediction_horizon=prediction_horizon)

            for batch_idx in range(actual_batch_size):
                inputs_b = index_dict_of_batched_vectors_tf(inputs, batch_idx)
                outputs_from_start_t_b = index_dict_of_batched_vectors_tf(outputs_from_start_t, batch_idx)
                predictions_from_start_t_b = index_dict_of_batched_vectors_tf(predictions_from_start_t, batch_idx)
                actions_b = actions_from_start_t[batch_idx]
                yield PredictionActualExample(inputs=inputs_b,
                                              outputs=outputs_from_start_t_b,
                                              actions=actions_b,
                                              predictions=predictions_from_start_t_b,
                                              prediction_start_t=prediction_start_t,
                                              labeling_params=labeling_params,
                                              actual_prediction_horizon=actual_prediction_horizon)


def generate_recovery_examples(fwd_model,
                               tf_dataset: tf.data.TFRecordDataset,
                               dataset: DynamicsDataset,
                               labeling_params: Dict):
    batch_size = 1
    for prediction_actual in one_step_prediction_actual_generator(fwd_model, tf_dataset, batch_size, dataset, labeling_params):
        yield from generate_recovery_actions_examples(prediction_actual)


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


def generate_recovery_actions_examples(prediction_actual: PredictionActualExample):
    inputs = prediction_actual.inputs
    outputs = prediction_actual.outputs
    predictions = prediction_actual.predictions
    start_t = prediction_actual.prediction_start_t
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

        # this slice gives arrays of fixed length (ex, 5)
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
        actions = prediction_actual.actions[action_slice]
        out_example['action'] = actions

        # compute label
        is_close = compute_is_close_tf(actual_states_dict=sliced_outputs,
                                       predicted_states_dict=sliced_predictions,
                                       labeling_params=labeling_params)
        is_close = tf.cast(is_close, dtype=tf.float32)
        out_example['is_close'] = is_close

        # TODO: to allow for show examples, yield so long as there's a "recovering" sequence from the start
        #  so for example if it looks like                  [-, 0, 0, 1, 1, 0, 0, 0] then
        #  still include that example, and the make will be [0, 1, 1, 1, 1, 0, 0, 0]
        print(is_close[1:])
        if remove_batch(is_recovering(add_batch(is_close[1:]))):
            import matplotlib.pyplot as plt
            anim = get_scenario('link_bot').animate_predictions_from_classifier_dataset(dataset_element=out_example,
                                                                                        state_keys=['link_bot'],
                                                                                        example_idx=0,
                                                                                        fps=5)
            plt.show()
            yield out_example


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
