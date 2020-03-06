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

        self.action_feature_names = []

        self.state_feature_names = []

        self.constant_feature_names = [
            'image',
            'label',
            # 'action',
            # 'full_env/env',
            # 'full_env/extent',
            # 'full_env/res',
            # 'label',
            # 'link_bot',
            # 'link_bot_next',
            # 'planned_state/link_bot',
            # 'planned_state/link_bot_next',
        ]

    def post_process(self, dataset: tf.data.TFRecordDataset, n_parallel_calls: int):

        @tf.function
        def _discard(const, action, state):
            return const

        dataset = dataset.map(_discard)

        return dataset
