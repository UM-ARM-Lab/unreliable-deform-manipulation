#!/usr/bin/env python
import argparse
import logging
import pathlib
import time
from typing import Dict

import colorama
import hjson
import tensorflow as tf
from colorama import Fore

import rospy
from arc_utilities import ros_init
from arc_utilities.path_utils import rm_tree
from link_bot_classifiers import train_test_classifier
from link_bot_data.classifier_dataset_utils import make_classifier_dataset_from_params_dict
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.pycommon import paths_from_json
from link_bot_pycommon.serialization import my_hdumps
from shape_completion_training.model.filepath_tools import get_trial_path


class TestThresholdSensitivity:

    def __init__(self, log: Dict, threshold: float):
        self.log = log
        self.threshold = threshold
        self.phase2_dataset_dir = pathlib.Path(self.log['phase_2_datset_dir'])
        self.udnn_model_dirs = paths_from_json(self.log['udnn_model_dirs'])
        subdir_name = 'scirob_dragging_test_threshold_sensitivity'
        self.classifier_dataset_base_dir = pathlib.Path('classifier_data') / subdir_name
        self.classifier_dataset_outdir = self.classifier_dataset_base_dir / f"threshold={self.threshold}"
        self.trials_directory = pathlib.Path('trials') / subdir_name

    def make_classifier_dataset(self, regenerate: bool, labeling_params: Dict, take: int):
        if 'classifier_dataset_dirs' in self.log:
            classifier_dataset_dirs = self.log['classifier_dataset_dirs']
            if self.classifier_dataset_outdir.as_posix() in classifier_dataset_dirs and not regenerate:
                print(Fore.YELLOW + f"dataset {self.classifier_dataset_outdir.as_posix()} already exists")
                return

        # we have to delete, otherwise if the new dataset is smaller, there will be old files left around
        if self.classifier_dataset_outdir.exists():
            rm_tree(self.classifier_dataset_outdir)
        self.classifier_dataset_outdir.mkdir(exist_ok=True, parents=True)
        make_classifier_dataset_from_params_dict(dataset_dir=self.phase2_dataset_dir,
                                                 fwd_model_dir=self.udnn_model_dirs,
                                                 batch_size=16,
                                                 labeling_params=labeling_params, outdir=self.classifier_dataset_outdir,
                                                 use_gt_rope=False, visualize=False, take=take)

        self.log['classifier_dataset_dirs'].append(self.classifier_dataset_outdir)
        rospy.loginfo(Fore.GREEN + self.classifier_dataset_outdir.as_posix())

    def learn_classifier(self, classifier_hparams_filename: pathlib.Path, batch_size: int, epochs: int, seed: int,
                         retrain: bool):
        self.trials_directory.mkdir(exist_ok=True, parents=True)
        group_name = f"threshold={self.threshold}"

        if not retrain:
            trial_path = get_trial_path(group_name=group_name, trials_directory=self.trials_directory)
            trial_dirs = self.log['trial_dirs']
            for trial_dir in trial_dirs:
                if str(self.threshold) in trial_dir:
                    trial_path = self.trials_directory / trial_dir
                    return trial_path

        train_test_classifier.train_main(dataset_dirs=[self.classifier_dataset_outdir],
                                         model_hparams=classifier_hparams_filename,
                                         log=group_name,
                                         trials_directory=self.trials_directory,
                                         checkpoint=None,
                                         batch_size=batch_size,
                                         epochs=epochs,
                                         use_gt_rope=False,
                                         seed=seed)
        return trial_path

    def evaluate(self, batch_size: int, trial_path: pathlib.Path):
        print(Fore.GREEN + f"Evaluating {trial_path.as_posix()}")
        train_metrics = train_test_classifier.eval_main(dataset_dirs=[self.classifier_dataset_outdir],
                                                        checkpoint=trial_path / 'best_checkpoint',
                                                        mode='train',
                                                        trials_directory=self.trials_directory,
                                                        batch_size=batch_size,
                                                        use_gt_rope=False,
                                                        )
        val_metrics = train_test_classifier.eval_main(dataset_dirs=[self.classifier_dataset_outdir],
                                                      checkpoint=trial_path / 'best_checkpoint',
                                                      mode='val',
                                                      trials_directory=self.trials_directory,
                                                      batch_size=batch_size,
                                                      use_gt_rope=False,
                                                      )
        output = {
            'threshold':              self.threshold,
            'classifier_dataset_dir': self.classifier_dataset_outdir,
            'udnn_model_dirs':        self.udnn_model_dirs,
            'train_metrics':          train_metrics,
            'validation_metrics':     val_metrics,
        }
        return output


def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('logfile', type=pathlib.Path)
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--take', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    ros_init.rospy_and_cpp_init(pathlib.Path(__file__).stem)

    with args.logfile.open("r") as logfile:
        log = hjson.load(logfile)

    classifier_hparams_filename = pathlib.Path('hparams/classifier/scirob_dragging_threshold_sensitivity.hjson')
    unique_name = f"metrics_{int(time.time())}"
    results_dir = pathlib.Path("results") / 'scirob_dragging_threshold_sensitivity'
    results_dir.mkdir(exist_ok=True)
    output_filename = results_dir / unique_name

    outputs = []
    for threshold in [0.025, 0.065, 0.105]:
        labeling_params = {
            # TODO: line search over thresholds
            'threshold':                     threshold,
            'classifier_horizon':            2,
            'start_step':                    4,
            'perception_reliability_method': 'gt',
        }

        tts = TestThresholdSensitivity(log, threshold)
        tts.make_classifier_dataset(regenerate=args.regenerate,
                                    labeling_params=labeling_params,
                                    take=args.take)
        trial_path = tts.learn_classifier(classifier_hparams_filename=classifier_hparams_filename,
                                          batch_size=args.batch_size,
                                          epochs=args.epochs,
                                          seed=0,
                                          retrain=args.retrain,
                                          )
        output = tts.evaluate(batch_size=args.batch_size, trial_path=trial_path)

        outputs.append(output)

    output_str = my_hdumps(outputs)
    print(Fore.GREEN + "Results:")
    print(output_str)
    print(Fore.GREEN + f"Saved {output_filename.as_posix()}")
    with output_filename.open("w") as output_file:
        output_file.write(output_str.encode("utf-8"))

    log_str = my_hdumps(log)
    with args.logfile.open("w") as logfile:
        logfile.write(log_str)

    ros_init.shutdown()


if __name__ == '__main__':
    main()
