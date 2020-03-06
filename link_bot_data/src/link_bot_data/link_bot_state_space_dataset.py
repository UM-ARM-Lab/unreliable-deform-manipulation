import pathlib
from typing import List

import tensorflow as tf

from link_bot_data.base_dataset import BaseDataset
from link_bot_planning.params import LocalEnvParams, FullEnvParams


class LinkBotStateSpaceDataset(BaseDataset):
    def __init__(self, dataset_dirs: List[pathlib.Path]):
        super(LinkBotStateSpaceDataset, self).__init__(dataset_dirs)

        self.local_env_params = LocalEnvParams.from_json(self.hparams['local_env_params'])
        self.full_env_params = FullEnvParams.from_json(self.hparams['full_env_params'])

        self.action_feature_names = ['%d/action']

        self.state_feature_names = [
            '%d/time_idx',
            '%d/traj_idx',
        ]

        self.states_description = self.hparams['states_description']
        for state_key in self.states_description.keys():
            self.state_feature_names.append('%d/{}'.format(state_key))

        self.constant_feature_names = [
            'full_env/env',
            'full_env/extent',
            'full_env/origin',
            'full_env/res',
        ]

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        # FIXME: don't separate const/state/action to begin with?
        def _combine_data(const_data, state_like_sequences, action_like_sequences):
            input_dict = {}
            for k, v in state_like_sequences.items():
                # chop off the last time step since that's not part of the input
                input_dict[k] = v[:-1]
            input_dict.update(action_like_sequences)
            input_dict.update(const_data)
            output_dict = {}
            output_dict.update(state_like_sequences)
            return input_dict, output_dict

        dataset = dataset.map(_combine_data, num_parallel_calls=n_parallel_calls)
        return dataset
