import pathlib
from typing import List

import tensorflow as tf

from link_bot_data.state_space_dataset import StateSpaceDataset
from link_bot_planning.params import LocalEnvParams


class LinkBotStateSpaceDataset(StateSpaceDataset):
    def __init__(self, dataset_dirs: List[pathlib.Path]):
        super(LinkBotStateSpaceDataset, self).__init__(dataset_dirs)

        self.state_like_names_and_shapes['state_s'] = '%d/state', (self.hparams['n_state'],)
        self.action_like_names_and_shapes['action_s'] = '%d/action', (2,)

        # local environment stuff
        self.local_env_params = LocalEnvParams.from_json(self.hparams['local_env_params'])

        local_env_shape = (self.local_env_params.h_rows, self.local_env_params.w_cols)

        self.state_like_names_and_shapes['resolution_s'] = '%d/res', (1,)
        self.state_like_names_and_shapes['actual_local_env_s/origin'] = '%d/actual_local_env/origin', (2,)
        self.state_like_names_and_shapes['actual_local_env_s/extent'] = '%d/actual_local_env/extent', (4,)
        self.state_like_names_and_shapes['actual_local_env_s/env'] = '%d/actual_local_env/env', local_env_shape

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        def _convert_full_sequence_to_input_and_output_sequences(const_data, state_like_sequences, action_like_sequences):
            # separates into x/y, where x is all time steps except the last, and y is all the time steps (including the first)
            # the first is included in y just because it makes it easier to visualize,
            # since you don't need to do any combination with the first time step or anything
            input_dict = {}
            for k, v in state_like_sequences.items():
                # chop off the last time step since that's not part of the input
                input_dict[k] = v[:-1]
            output_dict = {'output_states': state_like_sequences['state_s']}
            input_dict.update(action_like_sequences)
            input_dict.update(const_data)
            return input_dict, output_dict

        return dataset.map(_convert_full_sequence_to_input_and_output_sequences, num_parallel_calls=n_parallel_calls)
