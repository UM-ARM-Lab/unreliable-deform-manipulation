import pathlib
from typing import List

import tensorflow as tf

from link_bot_data.base_dataset import BaseDataset, make_name_singular
from link_bot_planning.params import LocalEnvParams, FullEnvParams


class ClassifierDataset(BaseDataset):

    def __init__(self, dataset_dirs: List[pathlib.Path]):
        super(ClassifierDataset, self).__init__(dataset_dirs)

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
            self.state_like_names_and_shapes.append('%d/state/{}'.format(k))

        self.trajectory_constant_names_and_shapes = [
            'full_env/origin',
            'full_env/extent',
            'full_env/env',
        ]

        # FIXME: test that this is working and correctly parsing
        print(self.state_like_names_and_shapes)

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):

        @tf.function
        def _convert_sequences_to_transitions(constant_data: dict, state_like_sequences: dict, action_like_sequences: dict):
            # Create a dict of lists, where keys are the features we want in each transition, and values are the data.
            # The first dimension of these values is what will be split up into different examples
            transitions = {}
            singular_state_like_names = []
            next_singular_state_like_names = []
            for feature_name in state_like_sequences.keys():
                feature_name_singular, next_feature_name_singular = make_name_singular(feature_name)
                singular_state_like_names.append((feature_name_singular, feature_name))
                next_singular_state_like_names.append((next_feature_name_singular, feature_name))
                transitions[next_feature_name_singular] = []
                transitions[feature_name_singular] = []

            singular_action_like_names = []
            for feature_name in action_like_sequences.keys():
                feature_name_singular, _ = make_name_singular(feature_name)
                transitions[feature_name_singular] = []
                singular_action_like_names.append((feature_name_singular, feature_name))

            for feature_name in constant_data.keys():
                transitions[feature_name] = []

            # Fill the transitions dictionary with the data from the sequences
            sequence_length = action_like_sequences['action_s'].shape[0]
            for transition_idx in range(sequence_length):
                for feature_name, feature_name_plural in singular_state_like_names:
                    transitions[feature_name].append(state_like_sequences[feature_name_plural][transition_idx])
                for next_feature_name, feature_name_plural in next_singular_state_like_names:
                    transitions[next_feature_name].append(state_like_sequences[feature_name_plural][transition_idx + 1])

                for feature_name_singular, feature_name_plural in singular_action_like_names:
                    transitions[feature_name_singular].append(action_like_sequences[feature_name_plural][transition_idx])

                for feature_name in constant_data.keys():
                    transitions[feature_name].append(constant_data[feature_name])

            return tf.data.Dataset.from_tensor_slices(transitions)

        @tf.function
        def _label_transitions(transition: dict):
            pre_transition_distance = tf.norm(transition['state'] - transition['planned_state'])
            post_transition_distance = tf.norm(transition['state_next'] - transition['planned_state_next'])

            pre_threshold = self.hparams['labeling']['pre_close_threshold']
            post_threshold = self.hparams['labeling']['post_close_threshold']

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
            if self.hparams['labeling']['discard_pre_far'] and not transition['pre_close']:
                return False
            return True

        dataset = dataset.flat_map(_convert_sequences_to_transitions)

        dataset = dataset.map(_label_transitions)

        dataset = dataset.filter(_filter_pre_far_transitions)
        return dataset
