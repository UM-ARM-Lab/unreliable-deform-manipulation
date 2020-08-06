#!/usr/bin/env python

import argparse
import json
import pathlib
import time
from typing import Dict

import rospkg

from link_bot_data.base_collect_dynamics_data import DataCollector
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_service_provider import get_service_provider
from state_space_dynamics import train_test

r = rospkg.RosPack()


class FullStackRunner:

    def __init__(self, full_stack_params: Dict):
        self.full_stack_params = full_stack_params
        self.nickname = full_stack_params['nickname']
        self.unique_nickname = f"{self.nickname}_{int(time.time())}"
        service_provider_name = full_stack_params['service_provider']
        self.service_provider = get_service_provider(service_provider_name)

    def collect_dynamics_data_1(self, seed: int):
        collect_dynamics_1 = self.full_stack_params['collect_dynamics_1']
        scenario = collect_dynamics_1['scenario']
        collect_dynamics_data_params_filename = pathlib.Path(collect_dynamics_1['params'])
        link_bot_data_path = pathlib.Path(r.get_path('link_bot_data'))
        full_collect_dynamics_data_params_filename = link_bot_data_path / collect_dynamics_data_params_filename

        with full_collect_dynamics_data_params_filename.open('r') as collect_dynamics_data_params_file:
            collect_dynamics_data_params = json.load(collect_dynamics_data_params_file)

        data_collector = DataCollector(scenario_name=scenario,
                                       service_provider=self.service_provider,
                                       params=collect_dynamics_data_params,
                                       seed=seed,
                                       verbose=0)
        files_dataset = data_collector.collect_data(n_trajs=collect_dynamics_1['n_trajs'], nickname=self.unique_nickname)
        files_dataset.split()
        return files_dataset.root_dir

    def learn_dynamics(self, seed: int, dynamics_dataset_dir: pathlib.Path):
        learn_dynamics_params = self.full_stack_params['train_dynamics']
        n_ensemble = learn_dynamics_params['n_ensemble']
        batch_size = learn_dynamics_params['batch_size']
        epochs = learn_dynamics_params['epochs']
        state_space_dynamics_path = pathlib.Path(r.get_path('state_space_dynamics'))
        forward_model_hparams = state_space_dynamics_path / learn_dynamics_params['forward_model_hparams']

        trial_paths = []
        for ensemble_idx in range(n_ensemble):
            trial_path = train_test.train_main(dataset_dirs=[dynamics_dataset_dir],
                                               model_hparams=forward_model_hparams,
                                               trials_directory=pathlib.Path('dy_trials'),
                                               checkpoint=None,
                                               log=self.unique_nickname,
                                               batch_size=batch_size,
                                               epochs=epochs,
                                               seed=seed,
                                               ensemble_idx=ensemble_idx)
            trial_paths.append(trial_path)

        return trial_paths

    def collect_dynamics_data_2(self, seed: int):
        pass

    def make_classifier_dataset(self):
        pass

    def learn_classifier(self):
        pass

    def make_recovery_dataset(self):
        pass

    def learn_recovery(self):
        pass

    def planning_evaluation(self):
        pass


def main():
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("full_stack_param", type=pathlib.Path)

    args = parser.parse_args()

    with args.full_stack_param.open('r') as f:
        full_stack_params = json.load(f)

    fsr = FullStackRunner(full_stack_params)
    seed = full_stack_params['seed']
    dynamics_dataset_dir = fsr.collect_dynamics_data_1(seed)
    fsr.learn_dynamics(seed, dynamics_dataset_dir)


if __name__ == '__main__':
    main()
