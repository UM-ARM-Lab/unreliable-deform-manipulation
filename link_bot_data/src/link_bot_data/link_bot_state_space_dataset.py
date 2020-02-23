import pathlib
from typing import List

import tensorflow as tf

from link_bot_data.base_dataset import BaseDataset
from link_bot_planning.params import LocalEnvParams, FullEnvParams


class LinkBotStateSpaceDataset(BaseDataset):
    def __init__(self, dataset_dirs: List[pathlib.Path]):
        super(LinkBotStateSpaceDataset, self).__init__(dataset_dirs)

        # local environment stuff
        self.local_env_params = LocalEnvParams.from_json(self.hparams['local_env_params'])
        self.full_env_params = FullEnvParams.from_json(self.hparams['full_env_params'])
        self.tether = self.hparams['tether']

        self.action_like_names_and_shapes = ['%d/action']

        self.state_like_names_and_shapes = [
            '%d/state/link_bot',
            '%d/state/local_env',
            '%d/state/local_env_origin',
            '%d/res',
            '%d/time_idx',
            '%d/traj_idx',
        ]
        if self.tether:
            self.state_like_names_and_shapes.append('%d/state/tether')

        self.trajectory_constant_names_and_shapes = [
            'full_env/origin',
            'full_env/extent',
            'full_env/env',
        ]

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        def _convert_full_sequence_to_input_and_output_sequences(const_data, state_like_sequences, action_like_sequences):
            # separates into x/y, where x is all time steps except the last, and y is all the time steps (including the first)
            # the first is included in y just because it makes it easier to visualize,
            # since you don't need to do any combination with the first time step or anything
            input_dict = {}
            for k, v in state_like_sequences.items():
                # chop off the last time step since that's not part of the input
                input_dict[k] = v[:-1]
            output_dict = {
                'link_bot': state_like_sequences['state/link_bot']
            }
            if self.tether:
                output_dict['tether'] = state_like_sequences['state/tether']
            input_dict.update(action_like_sequences)
            input_dict.update(const_data)
            return input_dict, output_dict

        dataset = dataset.map(_convert_full_sequence_to_input_and_output_sequences, num_parallel_calls=n_parallel_calls)
        return dataset
