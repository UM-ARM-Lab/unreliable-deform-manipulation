import pathlib
from typing import List

from link_bot_data.state_space_dataset import StateSpaceDataset
import tensorflow as tf
from link_bot_planning.params import LocalEnvParams, FullEnvParams


class TetherClassifierDataset(StateSpaceDataset):

    def __init__(self, dataset_dirs: List[pathlib.Path]):
        super(TetherClassifierDataset, self).__init__(dataset_dirs)

        self.full_env_params = FullEnvParams.from_json(self.hparams['full_env_params'])

        self.n_state = self.hparams['n_state']
        self.n_action = self.hparams['n_action']
        self.n_channels = self.n_state + self.n_action + 1
        self.full_env_shape = (self.full_env_params.h_rows, self.full_env_params.w_cols, self.n_channels)

        self.trajectory_constant_names_and_shapes['resolution_next'] = 'resolution_next', (1,)
        self.trajectory_constant_names_and_shapes['resolution'] = 'resolution', (1,)

        # These are the actual states -- should only be used for computing labels
        self.trajectory_constant_names_and_shapes['state_next'] = 'state_next', (self.n_state,)
        self.trajectory_constant_names_and_shapes['state'] = 'state', (self.n_state,)

        self.trajectory_constant_names_and_shapes['planned_state_next'] = 'planned_state_next', (self.n_state,)
        self.trajectory_constant_names_and_shapes['planned_state'] = 'planned_state', (self.n_state,)

        self.trajectory_constant_names_and_shapes['action'] = 'action', (self.n_action,)

        self.trajectory_constant_names_and_shapes['label'] = 'label', (1,)

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        @tf.function
        def _discard_seqs(trajectory_constant_data, _, __):
            return trajectory_constant_data

        dataset = dataset.map(_discard_seqs)
        return dataset
