import pathlib

import tensorflow as tf
from colorama import Fore

from link_bot_data.state_space_dataset import StateSpaceDataset
from link_bot_planning.params import LocalEnvParams, EnvParams, PlannerParams


class ClassifierDataset(StateSpaceDataset):

    def __init__(self,
                 dataset_dir: pathlib.Path,
                 is_labeled: bool = False):
        super(ClassifierDataset, self).__init__(dataset_dir)

        self.is_labeled = is_labeled
        if not self.is_labeled and 'labeled' in str(dataset_dir):
            print(Fore.YELLOW + "I noticed 'labeled' in the dataset path, so I will attempt to load labels" + Fore.RESET)
            self.is_labeled = True
        self.hparams['local_env_params'] = LocalEnvParams.from_json(self.hparams['local_env_params'])
        self.hparams['env_params'] = EnvParams.from_json(self.hparams['env_params'])
        self.hparams['planner_params'] = PlannerParams.from_json(self.hparams['planner_params'])

        local_env_shape = (self.hparams['local_env_params'].h_rows, self.hparams['local_env_params'].w_cols)

        self.action_like_names_and_shapes['actions'] = '%d/action', (2,)

        self.state_like_names_and_shapes['res'] = '%d/res', (1,)
        self.state_like_names_and_shapes['actual_local_envs/origin'] = '%d/actual_local_env/origin', (2,)
        self.state_like_names_and_shapes['actual_local_envs/extent'] = '%d/actual_local_env/extent', (4,)
        self.state_like_names_and_shapes['actual_local_envs/env'] = '%d/actual_local_env/env', local_env_shape
        self.state_like_names_and_shapes['planned_local_envs/extent'] = '%d/planned_local_env/extent', (4,)
        self.state_like_names_and_shapes['planned_local_envs/origin'] = '%d/planned_local_env/origin', (2,)
        self.state_like_names_and_shapes['planned_local_envs/env'] = '%d/planned_local_env/env', (2,)
        self.trajectory_constant_names_and_shapes['local_env_rows'] = 'local_env_rows', (1,)
        self.trajectory_constant_names_and_shapes['local_env_cols'] = 'local_env_cols', (1,)
        self.state_like_names_and_shapes['states'] = '%d/state', (self.hparams['n_state'],)
        # self.state_like_names_and_shapes['next_state'] = '%d/next_state', (self.hparams['n_state'],)
        self.state_like_names_and_shapes['actions'] = '%d/action', (2,)
        self.state_like_names_and_shapes['planned_states'] = '%d/planned_state', (self.hparams['n_state'],)
        # self.state_like_names_and_shapes['planned_next_state'] = '%d/planned_next_state', (self.hparams['n_state'],)

        # TODO: implement pairing up of states
