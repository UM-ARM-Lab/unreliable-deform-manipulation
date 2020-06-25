from typing import Dict

import numpy as np
import tensorflow as tf

from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import add_predicted, batch_tf_dataset
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import gather_dict, add_batch, remove_batch, index_dict_of_batched_vectors_tf
from state_space_dynamics.model_utils import EnsembleDynamicsFunction


class PredictionActualExample:
    def __init__(self,
                 example: Dict,
                 actual_states: Dict,
                 actions: Dict,
                 predictions: Dict,
                 start_t: int,
                 labeling_params: Dict,
                 actual_prediction_horizon: int,
                 batch_size: int):
        self.dataset_element = example
        self.actual_states = actual_states
        self.actions = actions
        self.predictions = predictions
        self.prediction_start_t = start_t
        self.labeling_params = labeling_params
        self.actual_prediction_horizon = actual_prediction_horizon
        self.batch_size = batch_size


def generate_classifier_examples(fwd_model: EnsembleDynamicsFunction,
                                 tf_dataset: tf.data.Dataset,
                                 dataset: DynamicsDataset,
                                 labeling_params: Dict):
    batch_size = 256
    classifier_horizon = labeling_params['classifier_horizon']
    scenario = fwd_model.scenario
    assert classifier_horizon >= 2
    tf_dataset = batch_tf_dataset(tf_dataset, batch_size, drop_remainder=False)

    for idx, _ in enumerate(tf_dataset):
        pass
    n_total_batches = idx

    for idx, example in enumerate(tf_dataset):
        print(f"{idx} / {n_total_batches}")
        actual_batch_size = int(example['traj_idx'].shape[0])

        for start_t in range(0, dataset.sequence_length - classifier_horizon + 1, labeling_params['start_step']):
            prediction_end_t = dataset.sequence_length
            actual_prediction_horizon = prediction_end_t - start_t
            actual_states_from_start_t = {k: example[k][:, start_t:prediction_end_t] for k in fwd_model.state_keys}
            actions_from_start_t = {k: example[k][:, start_t:prediction_end_t - 1] for k in fwd_model.action_keys}

            predictions_from_start_t = fwd_model.propagate_differentiable_batched(actual_states_from_start_t,
                                                                                  actions_from_start_t)
            prediction_actual = PredictionActualExample(example=example,
                                                        actions=actions_from_start_t,
                                                        actual_states=actual_states_from_start_t,
                                                        predictions=predictions_from_start_t,
                                                        start_t=start_t,
                                                        labeling_params=labeling_params,
                                                        actual_prediction_horizon=actual_prediction_horizon,
                                                        batch_size=actual_batch_size)
            yield from generate_classifier_examples_from_batch(scenario, prediction_actual)


def generate_classifier_examples_from_batch(scenario: ExperimentScenario, prediction_actual: PredictionActualExample):
    labeling_params = prediction_actual.labeling_params
    prediction_horizon = prediction_actual.actual_prediction_horizon
    classifier_horizon = labeling_params['classifier_horizon']

    for classifier_start_t in range(0, prediction_horizon - classifier_horizon + 1):
        classifier_end_t = classifier_start_t + classifier_horizon

        full_env = prediction_actual.dataset_element['env']
        full_env_origin = prediction_actual.dataset_element['origin']
        full_env_extent = prediction_actual.dataset_element['extent']
        full_env_res = prediction_actual.dataset_element['res']
        traj_idx = prediction_actual.dataset_element['traj_idx']
        prediction_start_t = prediction_actual.prediction_start_t
        prediction_start_t_batched = tf.cast(
            tf.stack([prediction_start_t] * prediction_actual.batch_size, axis=0), tf.float32)
        classifier_start_t_batched = tf.cast(
            tf.stack([classifier_start_t] * prediction_actual.batch_size, axis=0), tf.float32)
        classifier_end_t_batched = tf.cast(
            tf.stack([classifier_end_t] * prediction_actual.batch_size, axis=0), tf.float32)
        out_example = {
            'env': full_env,
            'origin': full_env_origin,
            'extent': full_env_extent,
            'res': full_env_res,
            'traj_idx': traj_idx,
            'prediction_start_t': prediction_start_t_batched,
            'classifier_start_t': classifier_start_t_batched,
            'classifier_end_t': classifier_end_t_batched,
        }

        # this slice gives arrays of fixed length (ex, 5) which must be null padded from out_example_end_idx onwards
        state_slice = slice(classifier_start_t, classifier_start_t + classifier_horizon)
        action_slice = slice(classifier_start_t, classifier_start_t + classifier_horizon - 1)
        sliced_actual = {}
        for key, actual_state_component in prediction_actual.actual_states.items():
            actual_state_component_sliced = actual_state_component[:, state_slice]
            out_example[key] = actual_state_component_sliced
            sliced_actual[key] = actual_state_component_sliced

        sliced_predictions = {}
        for key, prediction_component in prediction_actual.predictions.items():
            prediction_component_sliced = prediction_component[:, state_slice]
            out_example[add_predicted(key)] = prediction_component_sliced
            sliced_predictions[key] = prediction_component_sliced

        # action
        sliced_actions = {}
        for key, action_component in prediction_actual.actions.items():
            action_component_sliced = action_component[:, action_slice]
            out_example[key] = action_component_sliced
            sliced_actions[key] = action_component_sliced

        # compute label
        is_close = compute_is_close_tf(actual_states_dict=sliced_actual, predicted_states_dict=sliced_predictions,
                                       labeling_params=labeling_params)
        out_example['is_close'] = tf.cast(is_close, dtype=tf.float32)

        is_first_predicted_state_close = is_close[:, 0]
        valid_indices = tf.where(is_first_predicted_state_close)
        valid_indices = tf.squeeze(valid_indices, axis=1)
        # keep only valid_indices from every key in out_example...
        valid_out_example = gather_dict(out_example, valid_indices)

        def debug():
            # Visualize example
            from link_bot_classifiers.visualization import visualize_classifier_example_3d
            for batch_idx in range(prediction_actual.batch_size):
                visualize_classifier_example_3d(scenario=scenario,
                                                example=index_dict_of_batched_vectors_tf(out_example, batch_idx),
                                                n_time_steps=classifier_horizon)

        # debug()

        yield valid_out_example


def compute_is_close_tf(actual_states_dict: Dict, predicted_states_dict: Dict, labeling_params: Dict):
    state_key = labeling_params['state_key']
    labeling_states = tf.convert_to_tensor(actual_states_dict[state_key])
    labeling_predicted_states = tf.convert_to_tensor(predicted_states_dict[state_key])
    # TODO: use scenario to compute distance here?
    model_error = tf.linalg.norm(labeling_states - labeling_predicted_states, axis=-1)
    threshold = labeling_params['threshold']
    is_close = model_error < threshold

    return is_close


def compute_label_np(actual_states_dict: Dict, predicted_states_dict: Dict, labeling_params: Dict):
    is_close = remove_batch(compute_is_close_tf(*add_batch(actual_states_dict, predicted_states_dict), labeling_params))
    return is_close.numpy()
