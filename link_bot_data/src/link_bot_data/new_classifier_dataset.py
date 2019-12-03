import pathlib

from link_bot_data.state_space_dataset import StateSpaceDataset
import tensorflow as tf
from link_bot_planning.params import LocalEnvParams


class NewClassifierDataset(StateSpaceDataset):

    def __init__(self, dataset_dir: pathlib.Path):
        super(NewClassifierDataset, self).__init__(dataset_dir)

        self.local_env_params = LocalEnvParams.from_json(self.hparams['local_env_params'])

        local_env_shape = (self.local_env_params.h_rows, self.local_env_params.w_cols)
        n_state = self.hparams['fwd_model_hparams']['dynamics_dataset_hparams']['n_state']
        n_action = self.hparams['fwd_model_hparams']['dynamics_dataset_hparams']['n_action']

        self.trajectory_constant_names_and_shapes['resolution_next'] = 'resolution_next', (1,)
        self.trajectory_constant_names_and_shapes['resolution'] = 'resolution', (1,)

        self.trajectory_constant_names_and_shapes['actual_local_env_next/origin'] = 'actual_local_env_next/origin', (2,)
        self.trajectory_constant_names_and_shapes['actual_local_env/origin'] = 'actual_local_env/origin', (2,)
        self.trajectory_constant_names_and_shapes['actual_local_env_next/extent'] = 'actual_local_env_next/extent', (4,)
        self.trajectory_constant_names_and_shapes['actual_local_env/extent'] = 'actual_local_env/extent', (4,)
        self.trajectory_constant_names_and_shapes['actual_local_env_next/env'] = 'actual_local_env_next/env', local_env_shape
        self.trajectory_constant_names_and_shapes['actual_local_env/env'] = 'actual_local_env/env', local_env_shape

        self.trajectory_constant_names_and_shapes['planned_local_env_next/origin'] = 'planned_local_env_next/origin', (2,)
        self.trajectory_constant_names_and_shapes['planned_local_env/origin'] = 'planned_local_env/origin', (2,)
        self.trajectory_constant_names_and_shapes['planned_local_env_next/extent'] = 'planned_local_env_next/extent', (4,)
        self.trajectory_constant_names_and_shapes['planned_local_env/extent'] = 'planned_local_env/extent', (4,)
        self.trajectory_constant_names_and_shapes['planned_local_env_next/env'] = 'planned_local_env_next/env', local_env_shape
        self.trajectory_constant_names_and_shapes['planned_local_env/env'] = 'planned_local_env/env', local_env_shape

        # These are the actual states -- should only be used for computing labels
        self.trajectory_constant_names_and_shapes['state_next'] = 'state_next', (n_state,)
        self.trajectory_constant_names_and_shapes['state'] = 'state', (n_state,)

        self.trajectory_constant_names_and_shapes['planned_state_next'] = 'planned_state_next', (n_state,)
        self.trajectory_constant_names_and_shapes['planned_state'] = 'planned_state', (n_state,)

        self.trajectory_constant_names_and_shapes['action'] = 'action', (n_action,)

        self.trajectory_constant_names_and_shapes['local_env_rows'] = 'local_env_rows', (1,)
        self.trajectory_constant_names_and_shapes['local_env_cols'] = 'local_env_cols', (1,)

        self.trajectory_constant_names_and_shapes['pre_dist'] = 'pre_dist', (1,)
        self.trajectory_constant_names_and_shapes['post_dist'] = 'post_dist', (1,)
        self.trajectory_constant_names_and_shapes['pre_close'] = 'pre_close', (1,)

        self.trajectory_constant_names_and_shapes['label'] = 'label', (1,)

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        @tf.function
        def _discard_seqs(trajectory_constant_data, _, __):
            return trajectory_constant_data

        dataset = dataset.map(_discard_seqs)
        return dataset
