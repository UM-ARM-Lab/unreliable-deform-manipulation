import pathlib

import tensorflow as tf

from link_bot_data.state_space_dataset import StateSpaceDataset
from link_bot_planning.params import LocalEnvParams, EnvParams, PlannerParams


def make_name_singular(feature_name):
    if "/" in feature_name:
        base_feature_name, postfix = feature_name.split("/")
        if not base_feature_name.endswith("_s"):
            raise ValueError("time-indexed feature name {} doesn't end with _s ".format(base_feature_name))
        base_feature_name_singular = base_feature_name[:-2]
        feature_name_singular = "{}/{}".format(base_feature_name_singular, postfix)
        next_feature_name_singular = "{}_next/{}".format(base_feature_name_singular, postfix)
    else:
        base_feature_name = feature_name
        if not base_feature_name.endswith("_s"):
            raise ValueError("time-indexed feature name {} doesn't end with _s ".format(base_feature_name))
        base_feature_name_singular = base_feature_name[:-2]
        feature_name_singular = base_feature_name_singular
        next_feature_name_singular = "{}_next".format(base_feature_name_singular)
    return feature_name_singular, next_feature_name_singular


class ClassifierDataset(StateSpaceDataset):

    def __init__(self, dataset_dir: pathlib.Path):
        super(ClassifierDataset, self).__init__(dataset_dir)

        self.local_env_params = LocalEnvParams.from_json(self.hparams['local_env_params'])

        local_env_shape = (self.local_env_params.h_rows, self.local_env_params.w_cols)
        n_state = self.hparams['fwd_model_hparams']['dynamics_dataset_hparams']['n_state']
        n_action = self.hparams['fwd_model_hparams']['dynamics_dataset_hparams']['n_action']

        self.action_like_names_and_shapes['action_s'] = '%d/action', (n_action,)

        self.state_like_names_and_shapes['resolution_s'] = '%d/res', (1,)
        self.state_like_names_and_shapes['actual_local_env_s/origin'] = '%d/actual_local_env/origin', (2,)
        self.state_like_names_and_shapes['actual_local_env_s/extent'] = '%d/actual_local_env/extent', (4,)
        self.state_like_names_and_shapes['actual_local_env_s/env'] = '%d/actual_local_env/env', local_env_shape
        self.state_like_names_and_shapes['planned_local_env_s/extent'] = '%d/planned_local_env/extent', (4,)
        self.state_like_names_and_shapes['planned_local_env_s/origin'] = '%d/planned_local_env/origin', (2,)
        self.state_like_names_and_shapes['planned_local_env_s/env'] = '%d/planned_local_env/env', local_env_shape
        self.trajectory_constant_names_and_shapes['local_env_rows'] = 'local_env_rows', (1,)
        self.trajectory_constant_names_and_shapes['local_env_cols'] = 'local_env_cols', (1,)
        self.state_like_names_and_shapes['state_s'] = '%d/state', (n_state,)
        self.state_like_names_and_shapes['planned_state_s'] = '%d/planned_state', (n_state,)

    # this code is really unreadable
    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        def _convert_sequences_to_transitions(constant_data, state_like_sequences, action_like_sequences):
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

            sequence_length = action_like_sequences['action_s'].shape[0]
            for transition_idx in range(sequence_length - 1):
                for feature_name, feature_name_plural in singular_state_like_names:
                    transitions[feature_name].append(state_like_sequences[feature_name_plural][transition_idx])
                for next_feature_name, feature_name_plural in next_singular_state_like_names:
                    transitions[next_feature_name].append(state_like_sequences[feature_name_plural][transition_idx + 1])

                for feature_name_singular, feature_name_plural in singular_action_like_names:
                    transitions[feature_name_singular].append(action_like_sequences[feature_name_plural][transition_idx])

                for feature_name in constant_data.keys():
                    transitions[feature_name].append(constant_data[feature_name])

            return tf.data.Dataset.from_tensor_slices(transitions)

        def _label_transitions(transition):

            pre_transition_distance = tf.norm(transition['state'] - transition['planned_state'])
            post_transition_distance = tf.norm(transition['state_next'] - transition['planned_state_next'])

            pre_close = pre_transition_distance < self.hparams['labeling']['threshold']
            transition['pre_close'] = pre_close

            post_close = post_transition_distance < self.hparams['labeling']['threshold']
            transition['label'] = None  # yes this is necessary. You can't add a key to a dict inside a py_func conditionally
            if post_close:
                transition['label'] = tf.convert_to_tensor([1], dtype=tf.float32)
            else:
                transition['label'] = tf.convert_to_tensor([0], dtype=tf.float32)
            return transition

        def _filter_pre_far_transitions(transition):
            if self.hparams['labeling']['discard_pre_far'] and not transition['pre_close']:
                return False
            return True

        dataset = dataset.flat_map(_convert_sequences_to_transitions)
        dataset = dataset.map(_label_transitions)
        dataset = dataset.filter(_filter_pre_far_transitions)
        return dataset
