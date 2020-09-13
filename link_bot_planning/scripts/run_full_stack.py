#!/usr/bin/env python
import argparse
import logging
import pathlib
import re
import time
from typing import Dict, List, Optional

import colorama
import hjson
import rospkg
import tensorflow as tf
from colorama import Fore

import rospy
from link_bot_classifiers import train_test_classifier, train_test_recovery
from link_bot_data.base_collect_dynamics_data import DataCollector
from link_bot_data.classifier_dataset_utils import make_classifier_dataset_from_params_dict
from link_bot_data.recovery_actions_utils import make_recovery_dataset_from_params_dict
from link_bot_planning.planning_evaluation import planning_evaluation
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.get_service_provider import get_service_provider
from link_bot_pycommon.pycommon import paths_from_json
from link_bot_pycommon.serialization import MyHjsonEncoder
from state_space_dynamics import train_test

r = rospkg.RosPack()


class FullStackRunner:

    def __init__(self, full_stack_params: Dict, launch: Optional[bool] = False):
        self.launch = launch
        self.gui = True
        self.full_stack_params = full_stack_params
        self.nickname = full_stack_params['nickname']
        self.unique_nickname = f"{self.nickname}_{int(time.time())}"
        service_provider_name = full_stack_params['service_provider']
        self.service_provider = get_service_provider(service_provider_name)

    def collect_dynamics_data_1(self, runlog: Dict, seed: int):
        collect_dynamics_1 = self.full_stack_params['collect_dynamics_1']
        scenario = collect_dynamics_1['scenario']
        collect_dynamics_data_params_filename = pathlib.Path(collect_dynamics_1['params'])
        link_bot_data_path = pathlib.Path(r.get_path('link_bot_data'))
        full_collect_dynamics_data_params_filename = link_bot_data_path / collect_dynamics_data_params_filename

        with full_collect_dynamics_data_params_filename.open('r') as collect_dynamics_data_params_file:
            collect_dynamics_data_params = hjson.load(collect_dynamics_data_params_file)

        if self.launch:
            self.service_provider.launch(collect_dynamics_1, gui=self.gui)

        data_collector = DataCollector(scenario_name=scenario,
                                       service_provider=self.service_provider,
                                       params=collect_dynamics_data_params,
                                       seed=seed,
                                       verbose=0)
        dynamics_data_1_nickname = self.nickname + '_phase1'
        # this function will add a time stamp/git hash to the nickname
        files_dataset = data_collector.collect_data(n_trajs=collect_dynamics_1['n_trajs'],
                                                    nickname=dynamics_data_1_nickname)

        if self.launch:
            self.service_provider.kill()

        files_dataset.split()
        return {
            'dynamics_dataset_dir': files_dataset.root_dir,
        }

    def collect_dynamics_data_2(self, runlog: Dict, seed: int):
        collect_dynamics_2 = self.full_stack_params['collect_dynamics_2']
        scenario = collect_dynamics_2['scenario']
        collect_dynamics_data_params_filename = pathlib.Path(collect_dynamics_2['params'])
        link_bot_data_path = pathlib.Path(r.get_path('link_bot_data'))
        full_collect_dynamics_data_params_filename = link_bot_data_path / collect_dynamics_data_params_filename

        with full_collect_dynamics_data_params_filename.open('r') as collect_dynamics_data_params_file:
            collect_dynamics_data_params = hjson.load(collect_dynamics_data_params_file)

        if self.launch:
            self.service_provider.launch(collect_dynamics_2, gui=self.gui)

        data_collector = DataCollector(scenario_name=scenario,
                                       service_provider=self.service_provider,
                                       params=collect_dynamics_data_params,
                                       seed=seed,
                                       verbose=0)
        dynamics_data_2_nickname = self.nickname + '_phase2'
        files_dataset = data_collector.collect_data(n_trajs=collect_dynamics_2['n_trajs'],
                                                    nickname=dynamics_data_2_nickname)

        if self.launch:
            self.service_provider.kill()

        files_dataset.split()
        return {
            'dynamics_dataset_dir': files_dataset.root_dir,
        }

    def learn_dynamics(self, runlog: Dict, seed: int):
        dynamics_dataset_dir = pathlib.Path(runlog['collect_dynamics_data_1']['dynamics_dataset_dir'])
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

        # Use one of the models we trained to compute the 90th percentile on the validation set
        classifier_threshold = train_test.compute_classifier_threshold(dataset_dirs=[dynamics_dataset_dir],
                                                                       checkpoint=trial_paths[0] / 'best_checkpoint',
                                                                       mode='val',
                                                                       batch_size=batch_size)
        return {
            'model_dirs': trial_paths,
            'classifier_threshold': classifier_threshold
        }

    def learn_full_dynamics(self, runlog: Dict, seed: int):
        dynamics_dataset_2 = pathlib.Path(runlog['collect_dynamics_data_2']['dynamics_dataset_dir'])
        learn_dynamics_params = self.full_stack_params['learn_full_dynamics']
        n_ensemble = learn_dynamics_params['n_ensemble']
        batch_size = learn_dynamics_params['batch_size']
        epochs = learn_dynamics_params['epochs']
        state_space_dynamics_path = pathlib.Path(r.get_path('state_space_dynamics'))
        forward_model_hparams = state_space_dynamics_path / learn_dynamics_params['forward_model_hparams']

        trial_paths = []
        for ensemble_idx in range(n_ensemble):
            trial_path = train_test.train_main(dataset_dirs=[dynamics_dataset_2],
                                               model_hparams=forward_model_hparams,
                                               trials_directory=pathlib.Path('dy_trials'),
                                               checkpoint=None,
                                               log=self.unique_nickname + '_full',
                                               batch_size=batch_size,
                                               epochs=epochs,
                                               seed=seed,
                                               ensemble_idx=ensemble_idx)
            trial_paths.append(trial_path)

        return {
            'model_dirs': trial_paths,
        }

    def make_classifier_dataset(self, runlog: Dict, seed: int):
        dynamics_dataset_2 = pathlib.Path(runlog['collect_dynamics_data_2']['dynamics_dataset_dir'])
        udnn_model_dirs = paths_from_json(runlog['learn_dynamics']['model_dirs'])
        make_classifier_dataset_params = self.full_stack_params['make_classifier_dataset']
        labeling_params_filename = pathlib.Path(make_classifier_dataset_params['labeling_params'])
        with labeling_params_filename.open("r") as labeling_params_file:
            labeling_params = hjson.load(labeling_params_file)
        labeling_params['threshold'] = runlog['learn_dynamics']['classifier_threshold']

        classifier_data_dir = pathlib.Path('classifier_data')
        classifier_data_dir.mkdir(exist_ok=True)
        outdir = classifier_data_dir / self.unique_nickname
        outdir.mkdir(exist_ok=True, parents=False)
        rospy.loginfo(Fore.GREEN + outdir.as_posix())
        classifier_dataset_dir = make_classifier_dataset_from_params_dict(dataset_dir=dynamics_dataset_2,
                                                                          fwd_model_dir=udnn_model_dirs,
                                                                          labeling_params=labeling_params,
                                                                          outdir=outdir)
        rospy.loginfo(Fore.GREEN + outdir.as_posix())
        return {
            'classifier_dataset_dir': classifier_dataset_dir,
        }

    def learn_classifier(self, runlog: Dict, seed: int):
        classifier_dataset_dir = pathlib.Path(runlog['make_classifier_dataset']['classifier_dataset_dir'])
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
        return {
            'model_dir': trial_path,
        }

    def make_recovery_dataset(self, runlog: Dict, seed: int):
        dynamics_dataset_2 = pathlib.Path(runlog['collect_dynamics_data_2']['dynamics_dataset_dir'])
        udnn_model_dirs = paths_from_json(runlog['learn_dynamics']['model_dirs'])
        classifier_model_dir = pathlib.Path(runlog['learn_classifier']['model_dir'])
        make_recovery_dataset_params = self.full_stack_params['make_recovery_dataset']
        labeling_params_filename = pathlib.Path(make_recovery_dataset_params['labeling_params'])
        with labeling_params_filename.open("r") as labeling_params_file:
            labeling_params = hjson.load(labeling_params_file)
        labeling_params['threshold'] = runlog['learn_dynamics']['classifier_threshold']
        recovery_data_dir = pathlib.Path('recovery_data')
        recovery_data_dir.mkdir(exist_ok=True)
        outdir = pathlib.Path('recovery_data') / self.unique_nickname
        outdir.mkdir(parents=False, exist_ok=True)
        batch_size = make_recovery_dataset_params['batch_size']
        recovery_dataset_dir = make_recovery_dataset_from_params_dict(dataset_dir=dynamics_dataset_2,
                                                                      fwd_model_dir=udnn_model_dirs,
                                                                      classifier_model_dir=classifier_model_dir,
                                                                      batch_size=batch_size,
                                                                      labeling_params=labeling_params,
                                                                      outdir=outdir)
        rospy.loginfo(Fore.GREEN + recovery_dataset_dir.as_posix())
        return {
            'recovery_dataset_dir': recovery_dataset_dir,
        }

    def learn_recovery(self, runlog: Dict, seed: int):
        recovery_dataset_dir = pathlib.Path(runlog['make_recovery_dataset']['recovery_dataset_dir'])
        classifier_model_dir = pathlib.Path(runlog['learn_classifier']['model_dir'])
        learn_recovery_params = self.full_stack_params['learn_recovery']
        batch_size = learn_recovery_params['batch_size']
        epochs = learn_recovery_params['epochs']
        recovery_module_path = pathlib.Path(r.get_path('link_bot_classifiers'))
        classifier_model_dir = classifier_model_dir / 'best_checkpoint'
        recovery_hparams = recovery_module_path / learn_recovery_params['recovery_hparams']
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
        return {
            'model_dir': trial_path,
        }

    def planning_evaluation(self, runlog: Dict, seed: int):
        classifier_model_dir = pathlib.Path(runlog['learn_classifier']['model_dir'])
        udnn_model_dirs = paths_from_json(runlog['learn_dynamics']['model_dirs'])
        full_dynamics_model_dirs = paths_from_json(runlog['learn_full_dynamics']['model_dirs'])
        recovery_model_dir = pathlib.Path(runlog['learn_recovery']['model_dir'])
        planning_module_path = pathlib.Path(r.get_path('link_bot_planning'))
        planning_evaluation_params = self.full_stack_params["planning_evaluation"]
        test_scenes_dir = pathlib.Path(planning_evaluation_params["test_scenes_dir"])
        n_trials = planning_evaluation_params['n_trials']
        trials = list(range(n_trials))
        planners_params_common_filename = pathlib.Path(planning_evaluation_params['planners_params_common'])
        planners_params = self.make_planners_params(classifier_model_dir, full_dynamics_model_dirs, udnn_model_dirs,
                                                    planners_params_common_filename, planning_evaluation_params,
                                                    recovery_model_dir)

        if self.launch:
            self.service_provider.launch(planning_evaluation_params, gui=self.gui)

        root = planning_module_path / 'results' / self.nickname
        outdir = planning_evaluation(root=root,
                                     planners_params=planners_params,
                                     trials=trials,
                                     test_scenes_dir=test_scenes_dir,
                                     skip_on_exception=False)

        if self.launch:
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
                planner_params = hjson.load(planners_params_common_file)
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
                    'recovery_model_dir': link_bot_planning_path / 'recovery_trials' / 'random' / 'random',
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
    tf.get_logger().setLevel(logging.ERROR)

    colorama.init(autoreset=True)
    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("full_stack_param", type=pathlib.Path)
    parser.add_argument("--from-logfile", type=pathlib.Path)
    parser.add_argument("--gui", action='store_true')
    parser.add_argument("--launch", action='store_true')
    parser.add_argument("--steps", help="a comma separated list of steps to explicitly include, regardless of logfile")

    args = parser.parse_args()

    rospy.init_node("run_full_stack")

    with args.full_stack_param.open('r') as f:
        full_stack_params = hjson.load(f)

    fsr = FullStackRunner(full_stack_params, launch=args.launch)
    fsr.gui = args.gui
    included_steps = args.steps.split(",")

    if args.from_logfile:
        with args.from_logfile.open("r") as logfile:
            runlog = hjson.loads(logfile.read())
        logfile_name = args.from_logfile
    else:
        # create a logfile
        logfile_dir = pathlib.Path("results") / "log" / fsr.unique_nickname
        logfile_dir.mkdir(parents=True)
        logfile_name = logfile_dir / "logfile.hjson"
        logfile = logfile_name.open("w")
        runlog = {}

    seed = full_stack_params['seed']
    if 'collect_dynamics_data_1' not in runlog:
        collect_dynamics_data_1_out = fsr.collect_dynamics_data_1(runlog, seed)
        runlog['collect_dynamics_data_1'] = collect_dynamics_data_1_out
        with logfile_name.open("w") as logfile:
            hjson.dump(runlog, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())

    if 'collect_dynamics_data_2' not in runlog:
        collect_dynamics_data_2_out = fsr.collect_dynamics_data_2(runlog, seed)
        runlog['collect_dynamics_data_2'] = collect_dynamics_data_2_out
        with logfile_name.open("w") as logfile:
            hjson.dump(runlog, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())

    if 'learn_dynamics' not in runlog:
        learn_dynamics_out = fsr.learn_dynamics(runlog, seed)
        runlog['learn_dynamics'] = learn_dynamics_out
        with logfile_name.open("w") as logfile:
            hjson.dump(runlog, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())

    if 'learn_full_dynamics' not in runlog:
        learn_full_dynamics_out = fsr.learn_full_dynamics(runlog, seed)
        runlog['learn_full_dynamics'] = learn_full_dynamics_out
        with logfile_name.open("w") as logfile:
            hjson.dump(runlog, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())

    if 'make_classifier_dataset' not in runlog:
        make_classifier_dataset_out = fsr.make_classifier_dataset(runlog, seed)
        runlog['make_classifier_dataset'] = make_classifier_dataset_out
        with logfile_name.open("w") as logfile:
            hjson.dump(runlog, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())

    if 'learn_classifier' not in runlog:
        learn_classifier_out = fsr.learn_classifier(runlog, seed)
        runlog['learn_classifier'] = learn_classifier_out
        with logfile_name.open("w") as logfile:
            hjson.dump(runlog, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())

    if 'make_recovery_dataset' not in runlog:
        make_recovery_dataset_out = fsr.make_recovery_dataset(runlog, seed)
        runlog['make_recovery_dataset'] = make_recovery_dataset_out
        with logfile_name.open("w") as logfile:
            hjson.dump(runlog, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())

    if 'learn_recovery' not in runlog:
        learn_recovery_out = fsr.learn_recovery(runlog, seed)
        runlog['learn_recovery'] = learn_recovery_out
        with logfile_name.open("w") as logfile:
            hjson.dump(runlog, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())

    if 'planning_evaluation' not in runlog or 'planning_evaluation' in included_steps:
        planning_evaluation_out = fsr.planning_evaluation(runlog, seed)
        runlog['planning_evaluation'] = planning_evaluation_out
        with logfile_name.open("w") as logfile:
            hjson.dump(runlog, logfile, cls=MyHjsonEncoder)
        rospy.loginfo(Fore.GREEN + logfile_name.as_posix())


if __name__ == '__main__':
    main()
