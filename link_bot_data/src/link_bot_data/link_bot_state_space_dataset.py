import pathlib

from link_bot_data.state_space_dataset import StateSpaceDataset
from link_bot_planning.params import LocalEnvParams


class LinkBotStateSpaceDataset(StateSpaceDataset):
    def __init__(self, dataset_dir: pathlib.Path):
        super(LinkBotStateSpaceDataset, self).__init__(dataset_dir)

        self.state_like_names_and_shapes['states'] = '%d/state', (self.hparams['n_state'],)
        self.action_like_names_and_shapes['actions'] = '%d/action', (2,)

        # local environment stuff
        self.hparams['local_env_params'] = LocalEnvParams.from_json(self.hparams['local_env_params'])
        local_env_shape = (self.hparams['local_env_params'].h_rows, self.hparams['local_env_params'].w_cols)
        self.state_like_names_and_shapes['res'] = '%d/res', (1,)
        self.state_like_names_and_shapes['actual_local_env/origin'] = '%d/actual_local_env/origin', (2,)
        self.state_like_names_and_shapes['actual_local_env/extent'] = '%d/actual_local_env/extent', (4,)
        self.state_like_names_and_shapes['actual_local_env/env'] = '%d/actual_local_env/env', local_env_shape
