import pathlib
from typing import List, Dict

import tensorflow as tf

from link_bot_data.base_dataset import BaseDataset
from link_bot_data.link_bot_dataset_utils import add_next, convert_sequences_to_transitions
from link_bot_planning.params import LocalEnvParams, FullEnvParams


class ClassifierDataset(BaseDataset):

    def __init__(self, dataset_dirs: List[pathlib.Path], params: Dict):
        super(ClassifierDataset, self).__init__(dataset_dirs)

        self.labeling_params = params

        self.local_env_params = LocalEnvParams.from_json(self.hparams['local_env_params'])
        self.full_env_params = FullEnvParams.from_json(self.hparams['full_env_params'])

        actual_state_keys = self.hparams['actual_state_keys']
        planned_state_keys = self.hparams['planned_state_keys']

        self.action_feature_names = ['%d/action']

        self.state_feature_names = [
            '%d/res',
            '%d/time_idx',
            '%d/traj_idx',
        ]

        for k in actual_state_keys:
            self.state_feature_names.append('%d/{}'.format(k))

        for k in planned_state_keys:
            self.state_feature_names.append('%d/planned_state/{}'.format(k))

        self.constant_feature_names = [
            'full_env/origin',
            'full_env/extent',
            'full_env/env',
            'full_env/res',
        ]

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):

        @tf.function
        def _label_transitions(transition: dict):
            # FIXME: don't hard code this
            state_key = self.labeling_params['state_key']
            state_key_next = add_next(state_key)
            planned_state_key = 'planned_state/{}'.format(state_key)
            planned_state_key_next = add_next(planned_state_key)
            pre_transition_distance = tf.norm(transition[state_key] - transition[planned_state_key])
            post_transition_distance = tf.norm(transition[state_key_next] - transition[planned_state_key_next])

            pre_threshold = self.labeling_params['pre_close_threshold']
            post_threshold = self.labeling_params['post_close_threshold']

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
            if self.labeling_params['discard_pre_far'] and not transition['pre_close']:
                return False
            return True

        # At this point, the dataset consists of tuples (const_data, state_data, action_data)
        dataset = dataset.flat_map(convert_sequences_to_transitions)
        dataset = dataset.map(_label_transitions)
        dataset = dataset.filter(_filter_pre_far_transitions)

        return dataset
