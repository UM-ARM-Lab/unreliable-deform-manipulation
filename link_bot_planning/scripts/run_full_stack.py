#!/usr/bin/env python

import argparse
import json
import pathlib
import re
import time
from typing import Dict, List

import colorama
import rospkg
from colorama import Fore

import rospy
from link_bot_classifiers import train_test_classifier, train_test_recovery
from link_bot_data.base_collect_dynamics_data import DataCollector
from link_bot_data.classifier_dataset_utils import make_classifier_dataset
from link_bot_data.recovery_actions_utils import make_recovery_dataset
from link_bot_planning.planning_evaluation import planning_evaluation
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

        self.service_provider.launch(collect_dynamics_1)

        data_collector = DataCollector(scenario_name=scenario,
                                       service_provider=self.service_provider,
                                       params=collect_dynamics_data_params,
                                       seed=seed,
                                       verbose=0)
        dynamics_data_1_nickname = self.nickname + '_phase1'
        # this function will add a time stamp/git hash to the nickname
        files_dataset = data_collector.collect_data(n_trajs=collect_dynamics_1['n_trajs'],
                                                    nickname=dynamics_data_1_nickname)

        self.service_provider.kill()

        files_dataset.split()
        return files_dataset.root_dir

    def collect_dynamics_data_2(self, seed: int):
        collect_dynamics_2 = self.full_stack_params['collect_dynamics_2']
        scenario = collect_dynamics_2['scenario']
        collect_dynamics_data_params_filename = pathlib.Path(collect_dynamics_2['params'])
        link_bot_data_path = pathlib.Path(r.get_path('link_bot_data'))
        full_collect_dynamics_data_params_filename = link_bot_data_path / collect_dynamics_data_params_filename

        with full_collect_dynamics_data_params_filename.open('r') as collect_dynamics_data_params_file:
            collect_dynamics_data_params = json.load(collect_dynamics_data_params_file)

        self.service_provider.launch(collect_dynamics_2)

        data_collector = DataCollector(scenario_name=scenario,
                                       service_provider=self.service_provider,
                                       params=collect_dynamics_data_params,
                                       seed=seed,
                                       verbose=0)
        dynamics_data_2_nickname = self.nickname + '_phase2'
        files_dataset = data_collector.collect_data(n_trajs=collect_dynamics_2['n_trajs'],
                                                    nickname=dynamics_data_2_nickname)

        self.service_provider.kill()

        files_dataset.split()
        return files_dataset.root_dir

    def learn_dynamics(self, seed: int, dynamics_dataset_dir: pathlib.Path):
        learn_dynamics_params = self.full_stack_params['learn_dynamics']
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

    def learn_full_dynamics(self, seed: int, dynamics_dataset_dir: pathlib.Path):
        learn_dynamics_params = self.full_stack_params['learn_full_dynamics']
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
                                               log=self.unique_nickname + '_full',
                                               batch_size=batch_size,
                                               epochs=epochs,
                                               seed=seed,
                                               ensemble_idx=ensemble_idx)
            trial_paths.append(trial_path)

        return trial_paths

    def make_classifier_dataset(self, dynamics_dataset2, fwd_model_dirs: List):
        make_classifier_dataset_params = self.full_stack_params['make_classifier_dataset']
        labeling_params = pathlib.Path(make_classifier_dataset_params['labeling_params'])

        classifier_data_dir = pathlib.Path('classifier_data')
        classifier_data_dir.mkdir(exist_ok=True)
        outdir = classifier_data_dir / self.unique_nickname
        outdir.mkdir(exist_ok=True, parents=False)
        rospy.loginfo(Fore.GREEN + outdir.as_posix())
        classifier_dataset_dir = make_classifier_dataset(dataset_dir=dynamics_dataset2,
                                                         fwd_model_dir=fwd_model_dirs,
                                                         labeling_params=labeling_params,
                                                         outdir=outdir)
        rospy.loginfo(Fore.GREEN + outdir.as_posix())
        return classifier_dataset_dir

    def learn_classifier(self, classifier_dataset_dir: pathlib.Path, seed: int):
        learn_classifier_params = self.full_stack_params['learn_classifier']
        batch_size = learn_classifier_params['batch_size']
        epochs = learn_classifier_params['epochs']
        classifiers_module_path = pathlib.Path(r.get_path('link_bot_classifiers'))
        classifier_hparams = classifiers_module_path / learn_classifier_params['classifier_hparams']
        trial_path = train_test_classifier.train_main(dataset_dirs=[classifier_dataset_dir],
                                                      model_hparams=classifier_hparams,
                                                      log=self.unique_nickname,
                                                      trials_directory=pathlib.Path('cl_trials'),
                                                      checkpoint=None,
                                                      batch_size=batch_size,
                                                      epochs=epochs,
                                                      seed=seed)
        return trial_path

    def make_recovery_dataset(self, dynamics_dataset_dir2: pathlib.Path,
                              fwd_model_dirs: List[pathlib.Path],
                              classifier_model_dir: pathlib.Path):
        make_recovery_dataset_params = self.full_stack_params['make_recovery_dataset']
        labeling_params = pathlib.Path(make_recovery_dataset_params['labeling_params'])
        recovery_data_dir = pathlib.Path('recovery_data')
        recovery_data_dir.mkdir(exist_ok=True)
        outdir = pathlib.Path('recovery_data') / self.unique_nickname
        outdir.mkdir(parents=False, exist_ok=True)
        recovery_dataset_dir = make_recovery_dataset(dataset_dir=dynamics_dataset_dir2,
                                                     fwd_model_dir=fwd_model_dirs,
                                                     classifier_model_dir=classifier_model_dir,
                                                     batch_size=make_recovery_dataset_params['batch_size'],
                                                     labeling_params=labeling_params,
                                                     outdir=outdir)
        rospy.loginfo(Fore.GREEN + recovery_dataset_dir.as_posix())
        return recovery_dataset_dir

    def learn_recovery(self, recovery_dataset_dir: pathlib.Path, classifier_model_dir: pathlib.Path, seed: int):
        learn_recovery_params = self.full_stack_params['learn_recovery']
        batch_size = learn_recovery_params['batch_size']
        epochs = learn_recovery_params['epochs']
        recoverys_module_path = pathlib.Path(r.get_path('link_bot_classifiers'))
        classifier_model_dir = classifier_model_dir / 'best_checkpoint'
        recovery_hparams = recoverys_module_path / learn_recovery_params['recovery_hparams']
        trial_path = train_test_recovery.train_main(dataset_dirs=[recovery_dataset_dir],
                                                    classifier_checkpoint=classifier_model_dir,
                                                    model_hparams=recovery_hparams,
                                                    log=self.unique_nickname,
                                                    trials_directory=pathlib.Path('cl_trials'),
                                                    checkpoint=None,
                                                    batch_size=batch_size,
                                                    epochs=epochs,
                                                    seed=seed)
        rospy.loginfo(Fore.GREEN + trial_path.as_posix())
        return trial_path

    def planning_evaluation(self,
                            fwd_model_dirs: List[pathlib.Path],
                            full_dynamics_model_dirs: List[pathlib.Path],
                            classifier_model_dir: pathlib.Path,
                            recovery_model_dir: pathlib.Path):
        planning_module_path = pathlib.Path(r.get_path('link_bot_planning'))
        planning_evaluation_params = self.full_stack_params["planning_evaluation"]
        n_trials = planning_evaluation_params['n_trials']
        trials = list(range(n_trials))
        planners_params_common_filename = pathlib.Path(planning_evaluation_params['planners_params_common'])
        planners_params = self.make_planners_params(classifier_model_dir, full_dynamics_model_dirs, fwd_model_dirs,
                                                    planners_params_common_filename, planning_evaluation_params,
                                                    recovery_model_dir)

        self.service_provider.launch(planning_evaluation_params)

        root = planning_module_path / 'results' / self.nickname
        outdir = planning_evaluation(root=root,
                                     planners_params=planners_params,
                                     trials=trials,
                                     skip_on_exception=False)

        self.service_provider.kill()

        rospy.loginfo(Fore.GREEN + outdir.as_posix())
        return outdir

    def make_planners_params(self,
                             classifier_model_dir: pathlib.Path,
                             full_dynamics_model_dirs: List[pathlib.Path],
                             udnn_model_dirs: List[pathlib.Path],
                             planners_params_common_filename: pathlib.Path,
                             planning_evaluation_params: Dict,
                             recovery_model_dir: pathlib.Path):
        planners_params = []
        for method_name in planning_evaluation_params['methods']:
            with planners_params_common_filename.open('r') as planners_params_common_file:
                planner_params = json.load(planners_params_common_file)
            if method_name == "classifier":
                method_fwd_model_dirs = [d / 'best_checkpoint' for d in udnn_model_dirs]
                method_classifier_model_dir = [classifier_model_dir / 'best_checkpoint']
                recovery = {'use_recovery': False}
            elif method_name == "learned_recovery":
                method_fwd_model_dirs = [d / 'best_checkpoint' for d in udnn_model_dirs]
                method_classifier_model_dir = [classifier_model_dir / 'best_checkpoint']
                recovery = {
                    'recovery_model_dir': recovery_model_dir / 'best_checkpoint',
                    'use_recovery': True,
                }
            elif method_name == "random_recovery":
                method_fwd_model_dirs = [d / 'best_checkpoint' for d in udnn_model_dirs]
                method_classifier_model_dir = [classifier_model_dir / 'best_checkpoint']
                link_bot_planning_path = pathlib.Path(r.get_path('link_bot_planning'))
                recovery = {
                    'recovery_model_dir': link_bot_planning_path / 'recovery_trials' / 'random',
                    'use_recovery': True,
                }
            elif method_name == "no_classifier":
                method_fwd_model_dirs = [d / 'best_checkpoint' for d in udnn_model_dirs]
                method_classifier_model_dir = [pathlib.Path('cl_trials/none_baseline/none')]
                recovery = {'use_recovery': False}
            elif method_name == "full_dynamics":
                method_fwd_model_dirs = [d / 'best_checkpoint' for d in full_dynamics_model_dirs]
                method_classifier_model_dir = [pathlib.Path('cl_trials/none_baseline/none')]
                recovery = {'use_recovery': False}
            else:
                raise NotImplementedError(f"Method {method_name} not implemented")
            planner_params['fwd_model_dir'] = method_fwd_model_dirs
            planner_params['classifier_model_dir'] = method_classifier_model_dir
            planner_params['recovery'] = recovery
            table_nickname = re.split(r'[_\-\s]', method_name)
            # TODO: bad API design
            table_config = {
                "nickname": table_nickname,
                "classifier": 'TODO',
                "recovery": 'TODO',
                "dynamics": 'TODO',
            }
            planner_params['table_config'] = table_config
            planners_params.append((method_name, planner_params))
        return planners_params


def main():
    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("full_stack_param", type=pathlib.Path)

    args = parser.parse_args()

    rospy.init_node("run_full_stack")

    with args.full_stack_param.open('r') as f:
        full_stack_params = json.load(f)

    fsr = FullStackRunner(full_stack_params)
    seed = full_stack_params['seed']
    # dynamics_dataset_dir1 = fsr.collect_dynamics_data_1(seed)
    dynamics_dataset_dir1 = pathlib.Path("fwd_model_data/sim_dual_phase1_1599289280_3d78a1f2a3_5120")

    # dynamics_dataset_dir2 = fsr.collect_dynamics_data_2(seed)
    dynamics_dataset_dir2 = pathlib.Path("fwd_model_data/sim_dual_phase2_1599290755_3d78a1f2a3_2048")

    # fwd_model_dirs = fsr.learn_dynamics(seed, dynamics_dataset_dir1)

    # TODO: implement this
    # fwd_model_dirs = infer_ensemble_models('dy_trials/sim_dual_1599289247_0')
    fwd_model_dirs = [
        pathlib.Path('dy_trials/sim_dual_1599289247_0/September_05_06-32-09_3d78a1f2a3/'),
        pathlib.Path('dy_trials/sim_dual_1599289247_1/September_05_07-25-25_3d78a1f2a3/'),
        pathlib.Path('dy_trials/sim_dual_1599289247_2/September_05_08-18-41_3d78a1f2a3/'),
        pathlib.Path('dy_trials/sim_dual_1599289247_3/September_05_09-12-12_3d78a1f2a3/'),
        pathlib.Path('dy_trials/sim_dual_1599289247_4/September_05_10-05-52_3d78a1f2a3/'),
        pathlib.Path('dy_trials/sim_dual_1599289247_5/September_05_10-58-56_3d78a1f2a3/'),
        pathlib.Path('dy_trials/sim_dual_1599289247_6/September_05_11-52-30_3d78a1f2a3/'),
        pathlib.Path('dy_trials/sim_dual_1599289247_7/September_05_12-46-21_3d78a1f2a3/'),
    ]

    # full_dynamics_model_dirs = fsr.learn_full_dynamics(seed, dynamics_dataset_dir2)
    full_dynamics_model_dirs = [pathlib.Path("dy_trials/sim_dual_1599289247_full_0/September_05_13-44-00_3d78a1f2a3/")]

    # classifier_dataset_dir = fsr.make_classifier_dataset(dynamics_dataset_dir2, fwd_model_dirs)
    classifier_dataset_dir = pathlib.Path("classifier_data/sim_dual_1599344051")
    # classifier_model_dir = fsr.learn_classifier(classifier_dataset_dir, seed)
    classifier_model_dir = pathlib.Path("cl_trials/sim_dual_1599348269/September_05_19-24-29_8440e91e85")

    # recovery_dataset_dir = fsr.make_recovery_dataset(dynamics_dataset_dir2, fwd_model_dirs, classifier_model_dir)
    recovery_dataset_dir = pathlib.Path("recovery_data/sim_dual_1599358002")
    # recovery_model_dir = fsr.learn_recovery(recovery_dataset_dir, classifier_model_dir, seed)
    recovery_model_dir = pathlib.Path("recovery_trials/sim_dual_1599363170/September_05_23-32-51_8440e91e85")
    # recovery_model_dir = None

    fsr.planning_evaluation(fwd_model_dirs, full_dynamics_model_dirs, classifier_model_dir, recovery_model_dir)


if __name__ == '__main__':
    main()
