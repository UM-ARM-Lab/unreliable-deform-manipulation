import pathlib
from typing import List, Dict

import numpy as np
import tensorflow as tf

from link_bot_data.base_dataset import BaseDataset
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_data.link_bot_dataset_utils import add_planned, null_pad, null_previous_states, \
    balance, null_diverged
from link_bot_planning.params import FullEnvParams


def add_model_predictions(fwd_model, tf_dataset, dataset: DynamicsDataset, labeling_params: Dict):
    prediction_horizon = labeling_params['prediction_horizon']
    classifier_horizon = labeling_params['classifier_horizon']
    assert prediction_horizon <= dataset.desired_sequence_length
    batch_size = 2048
    for dataset_element in tf_dataset.batch(batch_size):
        inputs, outputs = dataset_element
        full_env = inputs['full_env/env']
        full_env_origin = inputs['full_env/origin']
        full_env_extent = inputs['full_env/extent']
        full_env_res = inputs['full_env/res']
        traj_idx = inputs['traj_idx']

        for prediction_start_t in range(0, dataset.max_sequence_length - prediction_horizon + 1, labeling_params['start_step']):
            prediction_end_t = prediction_start_t + prediction_horizon
            outputs_from_start_t = {k: v[:, prediction_start_t:prediction_end_t] for k, v in outputs.items()}

            predictions_from_start_t = predict_subsequence(states_description=dataset.states_description,
                                                           fwd_model=fwd_model,
                                                           dataset_element=dataset_element,
                                                           prediction_start_t=prediction_start_t,
                                                           prediction_horizon=prediction_horizon)

            for batch_idx in range(full_env.shape[0]):
                for classifier_start_t in range(0, prediction_horizon - classifier_horizon):
                    max_classifier_end_t = min(classifier_start_t + classifier_horizon, prediction_horizon)
                    out_example_end_idx = 1
                    for classifier_end_t in range(classifier_start_t + 1, max_classifier_end_t):

                        out_example = {
                            'full_env/env': full_env[batch_idx],
                            'full_env/origin': full_env_origin[batch_idx],
                            'full_env/extent': full_env_extent[batch_idx],
                            'full_env/res': full_env_res[batch_idx],
                            'traj_idx': tf.squeeze(traj_idx[batch_idx]),
                            'prediction_start_t': prediction_start_t,
                            'classifier_start_t': classifier_start_t,
                            'classifier_end_t': classifier_end_t,
                        }

                        # this slice gives arrays of fixed length (ex, 5) which must be null padded from classifier_end_t onwards
                        classifier_full_slice = slice(classifier_start_t, classifier_start_t + classifier_horizon)
                        for name, output in outputs_from_start_t.items():
                            output_from_cst = output[batch_idx][classifier_full_slice]
                            null_padded_sequence = null_pad(output_from_cst, end=out_example_end_idx)
                            out_example[name] = null_padded_sequence

                        for name, prediction in predictions_from_start_t.items():
                            pred_from_cst = prediction[batch_idx][classifier_full_slice]
                            null_padded_sequence = null_pad(pred_from_cst, end=out_example_end_idx)
                            out_example[add_planned(name)] = null_padded_sequence

                        # action
                        out_example['action'] = inputs['action'][batch_idx]

                        # compute label
                        state_key = labeling_params['state_key']
                        planned_state_key = add_planned(state_key)

                        labeling_state = out_example[state_key][out_example_end_idx]
                        labeling_planned_state = out_example[planned_state_key][out_example_end_idx]

                        label = compute_label(labeling_params, labeling_state, labeling_planned_state)
                        out_example['label'] = label

                        yield out_example

                        out_example_end_idx += 1


def compute_label(labeling_params, labeling_state, labeling_planned_state):
    post_transition_distance = tf.norm(labeling_state - labeling_planned_state)
    threshold = labeling_params['threshold']
    post_close = post_transition_distance < threshold
    label = tf.expand_dims(tf.cast(post_close, dtype=tf.float32), axis=0)
    return label


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


class ClassifierDataset(BaseDataset):

    def __init__(self, dataset_dirs: List[pathlib.Path], no_balance=False):
        super(ClassifierDataset, self).__init__(dataset_dirs)
        self.no_balance = no_balance
        self.full_env_params = FullEnvParams.from_json(self.hparams['full_env_params'])
        self.labeling_params = self.hparams['labeling_params']
        self.label_state_key = self.hparams['labeling_params']['state_key']

        self.state_keys = self.hparams['state_keys']

        self.feature_names = [
            'full_env/env',
            'full_env/origin',
            'full_env/extent',
            'full_env/res',
            'traj_idx',
            'prediction_start_t',
            'classifier_start_t',
            'classifier_end_t',
            'action',
            'label',
        ]

        for k in self.hparams['states_description'].keys():
            self.feature_names.append(k)

        for k in self.state_keys:
            self.feature_names.append(add_planned(k))

        self.feature_names.append(add_planned('stdev'))

    def make_features_description(self):
        features_description = {}
        for feature_name in self.feature_names:
            features_description[feature_name] = tf.io.FixedLenFeature([], tf.string)

        return features_description

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        if not self.no_balance:
            dataset = balance(dataset, self.labeling_params)
        return dataset
