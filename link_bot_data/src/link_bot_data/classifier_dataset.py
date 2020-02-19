import pathlib
from enum import auto
from typing import List, Dict

import tensorflow as tf

from link_bot_data.base_dataset import BaseDataset
from link_bot_planning.params import LocalEnvParams, FullEnvParams
from link_bot_pycommon import args
from link_bot_pycommon.link_bot_sdf_utils import compute_extent


class ClassifierDatasetType(args.ArgsEnum):
    TRANSITION = auto()
    TRAJECTORY = auto()
    TRANSITION_IMAGE = auto()
    TRAJECTORY_IMAGE = auto()


def add_next(feature_name):
    if '/' in feature_name:
        prefix, suffix = feature_name.split('/')
        return prefix + '_next/' + suffix
    return feature_name + '_next'


@tf.function
def convert_sequences_to_transitions(constant_data: dict, state_like_sequences: dict, action_like_sequences: dict):
    # Create a dict of lists, where keys are the features we want in each transition, and values are the data.
    # The first dimension of these values is what will be split up into different examples
    transitions = {}
    state_like_names = []
    next_state_like_names = []
    for feature_name in state_like_sequences.keys():
        next_feature_name = add_next(feature_name)
        state_like_names.append(feature_name)
        next_state_like_names.append((next_feature_name, feature_name))
        transitions[feature_name] = []
        transitions[feature_name + "_all"] = []
        transitions[feature_name + "_all_stop"] = []
        transitions[next_feature_name] = []

    action_like_names = []
    for feature_name in action_like_sequences.keys():
        transitions[feature_name] = []
        transitions[feature_name + "_all"] = []
        transitions[feature_name + "_all_stop"] = []
        action_like_names.append(feature_name)

    for feature_name in constant_data.keys():
        transitions[feature_name] = []

    def _zero_pad_sequence(sequence, transition_idx):
        if transition_idx + 1 < sequence.shape[0]:
            sequence[transition_idx + 1:] = -1
        return sequence

    # Fill the transitions dictionary with the data from the sequences
    sequence_length = action_like_sequences['action'].shape[0]
    for transition_idx in range(sequence_length):
        for feature_name in state_like_names:
            transitions[feature_name].append(state_like_sequences[feature_name][transition_idx])
            # include all data up, zeroing out the future data
            zps = tf.numpy_function(_zero_pad_sequence, [state_like_sequences[feature_name], transition_idx], tf.float32)
            zps.set_shape(state_like_sequences[feature_name].shape)
            transitions[feature_name + '_all'].append(zps)
            transitions[feature_name + '_all_stop'].append(transition_idx + 1)
        for next_feature_name, feature_name in next_state_like_names:
            transitions[next_feature_name].append(state_like_sequences[feature_name][transition_idx + 1])

        for feature_name in action_like_names:
            transitions[feature_name].append(action_like_sequences[feature_name][transition_idx])
            # include all data up, zeroing out the future data
            zps = tf.numpy_function(_zero_pad_sequence, [action_like_sequences[feature_name], transition_idx], tf.float32)
            zps.set_shape(action_like_sequences[feature_name].shape)
            transitions[feature_name + '_all'].append(zps)
            transitions[feature_name + '_all_stop'].append(transition_idx + 1)

        for feature_name in constant_data.keys():
            transitions[feature_name].append(constant_data[feature_name])

    transition_dataset = tf.data.Dataset.from_tensor_slices(transitions)
    return transition_dataset


class ClassifierDataset(BaseDataset):

    def __init__(self, dataset_dirs: List[pathlib.Path], params: Dict):
        super(ClassifierDataset, self).__init__(dataset_dirs)

        self.classifier_dataset_params = params

        self.local_env_params = LocalEnvParams.from_json(self.hparams['local_env_params'])
        self.full_env_params = FullEnvParams.from_json(self.hparams['full_env_params'])

        actual_state_keys = self.hparams['actual_state_keys']
        planned_state_keys = self.hparams['planned_state_keys']

        self.action_like_names_and_shapes = ['%d/action']

        self.state_like_names_and_shapes = [
            '%d/res',
            '%d/time_idx',
            '%d/traj_idx',
        ]

        for k in actual_state_keys:
            self.state_like_names_and_shapes.append('%d/state/{}'.format(k))

        for k in planned_state_keys:
            self.state_like_names_and_shapes.append('%d/planned_state/{}'.format(k))

        self.trajectory_constant_names_and_shapes = [
            'full_env/origin',
            'full_env/extent',
            'full_env/env',
        ]

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):

        @tf.function
        def _label_transitions(transition: dict):
            pre_transition_distance = tf.norm(transition['state/link_bot'] - transition['planned_state/link_bot'])
            post_transition_distance = tf.norm(transition['state_next/link_bot'] - transition['planned_state_next/link_bot'])

            pre_threshold = self.classifier_dataset_params['pre_close_threshold']
            post_threshold = self.classifier_dataset_params['post_close_threshold']

            pre_close = pre_transition_distance < pre_threshold
            post_close = post_transition_distance < post_threshold

            # You're not allowed to modify input arguments, so we create a new dict and copy everything
            new_transition = {}
            for k, v in transition.items():
                new_transition[k] = v
            new_transition['pre_dist'] = pre_transition_distance
            new_transition['post_dist'] = post_transition_distance
            new_transition['pre_close'] = pre_close

            new_transition['label'] = None  # yes this is necessary. You can't add a key to a dict inside a py_func conditionally
            if post_close:
                new_transition['label'] = tf.convert_to_tensor([1], dtype=tf.float32)
            else:
                new_transition['label'] = tf.convert_to_tensor([0], dtype=tf.float32)
            return new_transition

        def _filter_pre_far_transitions(transition):
            if self.classifier_dataset_params['discard_pre_far'] and not transition['pre_close']:
                return False
            return True

        def _compute_extent(transition):
            res = transition['res']
            res_2d = tf.tile(tf.expand_dims(res, axis=0), [2])
            origin = transition['state/local_env_origin']
            next_origin = transition['state_next/local_env_origin']
            planned_origin = transition['planned_state/local_env_origin']
            planned_next_origin = transition['planned_state_next/local_env_origin']
            w_cols, h_rows = transition['state/local_env'].shape

            extent = tf.numpy_function(compute_extent, inp=[h_rows, w_cols, res_2d, origin], Tout=tf.float32)
            extent.set_shape([4])
            transition['state/local_env_extent'] = extent

            next_extent = tf.numpy_function(compute_extent, inp=[h_rows, w_cols, res_2d, next_origin], Tout=tf.float32)
            next_extent.set_shape([4])
            transition['state_next/local_env_extent'] = next_extent

            planned_extent = tf.numpy_function(compute_extent, inp=[h_rows, w_cols, res_2d, planned_origin], Tout=tf.float32)
            planned_extent.set_shape([4])
            transition['planned_state/local_env_extent'] = planned_extent

            planned_next_extent = tf.numpy_function(compute_extent, inp=[h_rows, w_cols, res_2d, planned_next_origin],
                                                    Tout=tf.float32)
            planned_next_extent.set_shape([4])
            transition['planned_state_next/local_env_extent'] = planned_next_extent
            return transition

        # At this point, the dataset consists of tuples (const_data, state_data, action_data)
        # if self.classifier_dataset_params['type'] == 'transition':
        dataset = dataset.flat_map(convert_sequences_to_transitions)
        dataset = dataset.map(_label_transitions)
        dataset = dataset.filter(_filter_pre_far_transitions)

        dataset = dataset.map(_compute_extent)

        return dataset
