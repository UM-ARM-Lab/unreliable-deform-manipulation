import pathlib
from typing import List

from link_bot_data.base_dataset import BaseDataset
import tensorflow as tf
from link_bot_planning.params import LocalEnvParams


class ImageClassifierDataset(BaseDataset):

    def __init__(self, dataset_dirs: List[pathlib.Path]):
        super(ImageClassifierDataset, self).__init__(dataset_dirs)

        self.local_env_params = LocalEnvParams.from_json(self.hparams['local_env_params'])

        self.n_state = self.hparams['n_state']
        self.n_action = self.hparams['n_action']
        self.n_channels = self.n_state + self.n_action + 1
        local_env_shape = (self.local_env_params.h_rows, self.local_env_params.w_cols, self.n_channels)
        self.trajectory_constant_names_and_shapes['image'] = 'image', local_env_shape

        self.trajectory_constant_names_and_shapes['label'] = 'label', (1,)

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):
        @tf.function
        def _discard_seqs(trajectory_constant_data, _, __):
            return trajectory_constant_data

        dataset = dataset.map(_discard_seqs)
        return dataset
