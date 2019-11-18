import itertools
import os
import re

from .state_space_dataset import StateSpaceDataset


class LinkBotStateSpaceDataset(StateSpaceDataset):
    def __init__(self, *args, **kwargs):
        super(LinkBotStateSpaceDataset, self).__init__(*args, **kwargs)

        self.state_like_names_and_shapes['state'] = '%d/state', (self.hparams.rope_config_dim,)
        self.action_like_names_and_shapes['actions'] = '%d/action', (2,)

        # local environment stuff
        self.state_like_names_and_shapes['res'] = '%d/res', (1,)
        self.state_like_names_and_shapes['actual_local_env/origin'] = '%d/actual_local_env/origin', (2,)
        self.state_like_names_and_shapes['actual_local_env/extent'] = '%d/actual_local_env/extent', (4,)
        self.state_like_names_and_shapes['actual_local_env/env'] = '%d/actual_local_env/env', (
            self.hparams.local_env_rows, self.hparams.local_env_cols)

        self._infer_seq_length_and_setup()

    def get_default_hparams_dict(self):
        default_hparams = super(LinkBotStateSpaceDataset, self).get_default_hparams_dict()
        hparams = dict(
            sequence_length=12,
            free_space_only=False,
            local_env_rows=50,
            local_env_cols=50,
            rope_config_dim=6,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def parser(self, serialized_example):
        state_like_seqs, action_like_seqs = super(LinkBotStateSpaceDataset, self).parser(serialized_example)
        return state_like_seqs, action_like_seqs

    def num_examples_per_epoch(self):
        # extract information from filename to count the number of trajectories in the dataset
        count = 0
        for filename in self.filenames:
            match = re.search('traj_(\d+)_to_(\d+).tfrecords', os.path.basename(filename))
            start_traj_iter = int(match.group(1))
            end_traj_iter = int(match.group(2))
            count += end_traj_iter - start_traj_iter + 1

        # alternatively, the dataset size can be determined like this, but it's very slow
        # count = sum(sum(1 for _ in tf.python_io.tf_record_iterator(filename)) for filename in filenames)
        return count
