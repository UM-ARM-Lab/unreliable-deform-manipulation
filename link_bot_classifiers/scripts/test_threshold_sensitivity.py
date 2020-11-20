#!/usr/bin/env python
import argparse
import logging
import pathlib
from typing import Dict, Optional

import colorama
import hjson
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Fore

import rospy
from arc_utilities import ros_init, dict_tools
from arc_utilities.path_utils import rm_tree
from link_bot_classifiers import train_test_classifier
from link_bot_data.classifier_dataset_utils import make_classifier_dataset_from_params_dict
from link_bot_pycommon.args import my_formatter
from link_bot_pycommon.matplotlib_utils import save_unconstrained_layout
from link_bot_pycommon.pycommon import paths_from_json
from link_bot_pycommon.serialization import my_hdumps
from moonshine.moonshine_utils import numpify


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

    def learn_classifier(self,
                         classifier_hparams_filename: pathlib.Path,
                         batch_size: int,
                         epochs: int,
                         seed: int,
                         take: Optional[int],
                         retrain: bool):
        self.trials_directory.mkdir(exist_ok=True, parents=True)
        group_name = f"threshold={self.threshold}_{seed}"

        print(Fore.GREEN + f"Training {group_name}")
        trial_path, _ = train_test_classifier.train_main(dataset_dirs=[self.classifier_dataset_outdir],
                                                         model_hparams=classifier_hparams_filename,
                                                         log=group_name,
                                                         trials_directory=self.trials_directory,
                                                         checkpoint=None,
                                                         validate=False,
                                                         batch_size=batch_size,
                                                         epochs=epochs,
                                                         use_gt_rope=False,
                                                         take=take,
                                                         seed=seed)
        return trial_path

    def evaluate(self, batch_size: int, trial_path: pathlib.Path, take: int):
        print(Fore.GREEN + f"Evaluating {trial_path.as_posix()}")
        train_metrics = train_test_classifier.eval_main(dataset_dirs=[self.classifier_dataset_outdir],
                                                        checkpoint=trial_path / 'best_checkpoint',
                                                        mode='train',
                                                        trials_directory=self.trials_directory,
                                                        batch_size=batch_size,
                                                        use_gt_rope=False,
                                                        take=take,
                                                        )
        val_metrics = train_test_classifier.eval_main(dataset_dirs=[self.classifier_dataset_outdir],
                                                      checkpoint=trial_path / 'best_checkpoint',
                                                      mode='val',
                                                      trials_directory=self.trials_directory,
                                                      batch_size=batch_size,
                                                      use_gt_rope=False,
                                                      take=take
                                                      )
        train_metrics = dict_tools.dict_round(numpify(train_metrics))
        val_metrics = dict_tools.dict_round(numpify(val_metrics))
        return train_metrics, val_metrics


def trending_boxplots(fig, ax, x, y, color, width, label):
    """ x is a vector, and x_i has an associaed data *column* vector y_i """
    ax.plot(x, np.mean(y, axis=0), label=label, c=color, zorder=2)
    ax.boxplot(y,
               positions=x,
               widths=width,
               patch_artist=True,
               boxprops=dict(facecolor='#00000000', color=color),
               capprops=dict(color=color),
               whiskerprops=dict(color=color),
               flierprops=dict(color=color, markeredgecolor=color),
               medianprops=dict(color=color),
               zorder=1,
               )


def plot(outputs: Dict, results_dir: pathlib.Path):
    thresholds = []
    train_accuracies = []
    val_accuracies = []
    for threshold, output in outputs.items():
        thresholds.append(float(threshold))
        train_accuracy = [m['accuracy'] for m in output['train_metrics']]
        train_accuracies.append(train_accuracy)
        val_accuracy = [m['accuracy'] for m in output['validation_metrics']]
        val_accuracies.append(val_accuracy)

    train_accuracies = np.array(train_accuracies).T
    val_accuracies = np.array(val_accuracies).T

    fig, axes = plt.subplots(1, 2)

    trending_boxplots(fig, axes[0], thresholds, train_accuracies, 'r', 0.02, 'train')
    trending_boxplots(fig, axes[1], thresholds, val_accuracies, 'b', 0.02, 'val')

    for ax in axes:
        ax.set_xlabel("threshold")
        ax.set_ylabel("accuracy")
        ax.set_xlim([-0.05, 0.2])
        ax.legend()

    filename = results_dir / 'plot.png'
    save_unconstrained_layout(fig, filename)
    plt.show()


def generate_outputs(args, classifier_hparams_filename, log, output_filename, outputs: Dict):
    if 'trial_dirs' in log:
        trial_dirs = log['trial_dirs']
    else:
        trial_dirs = {}

    for threshold in [0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105]:
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

        threshold_key = str(threshold)
        if threshold_key not in trial_dirs:
            trial_dirs[threshold_key] = {}

        for seed in range(3):
            seed_key = str(seed)
            if seed_key not in trial_dirs[threshold_key]:
                trial_dir_for_seed = tts.learn_classifier(classifier_hparams_filename,
                                                          batch_size=args.batch_size,
                                                          epochs=args.epochs,
                                                          seed=seed,
                                                          take=args.take,
                                                          retrain=args.retrain,
                                                          )
                trial_dirs[threshold_key][seed_key] = trial_dir_for_seed
                log['trial_dirs'] = trial_dirs
                save_log(args.logfile, log)
            else:
                print(f"Model {trial_dirs[threshold_key][seed_key]} for seed {seed} already exists")

        if threshold_key not in outputs:
            train_metrics = []
            val_metrics = []
            for trial_dir_for_seed in trial_dirs[threshold_key].values():
                train_metrics_for_seed, val_metrics_for_seed = tts.evaluate(batch_size=args.batch_size,
                                                                            trial_path=pathlib.Path(trial_dir_for_seed),
                                                                            take=args.take)
                train_metrics.append(train_metrics_for_seed)
                val_metrics.append(val_metrics_for_seed)

            output = {
                'threshold':              tts.threshold,
                'classifier_dataset_dir': tts.classifier_dataset_outdir,
                'udnn_model_dirs':        tts.udnn_model_dirs,
                'train_metrics':          train_metrics,
                'validation_metrics':     val_metrics,
            }
            outputs[threshold_key] = output

        output_str = my_hdumps(outputs)
        print(Fore.GREEN + "Results:")
        print(output_str)
        print(Fore.GREEN + f"Saved {output_filename.as_posix()}")
        with output_filename.open("w") as output_file:
            output_file.write(output_str)

        log['output_filename'] = output_filename
        save_log(args.logfile, log)

    return outputs


def save_log(logfilename, log):
    log_str = my_hdumps(log)
    with logfilename.open("w") as logfile:
        logfile.write(log_str)


def main():
    colorama.init(autoreset=True)
    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('logfile', type=pathlib.Path)
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--take', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=8)

    args = parser.parse_args()

    ros_init.rospy_and_cpp_init(pathlib.Path(__file__).stem)

    with args.logfile.open("r") as logfile:
        log = hjson.load(logfile)

    classifier_hparams_filename = pathlib.Path('hparams/classifier/scirob_dragging_threshold_sensitivity.hjson')
    results_dir = pathlib.Path("results") / 'scirob_dragging_threshold_sensitivity'
    results_dir.mkdir(exist_ok=True)
    output_filename = results_dir / 'metrics.hjson'

    if 'output_filename' in log:
        output_filename = pathlib.Path(log['output_filename'])
        with output_filename.open("r") as output_file:
            outputs = hjson.load(output_file)
    else:
        outputs = {}

    outputs = generate_outputs(args, classifier_hparams_filename, log, output_filename, outputs)
    log['output_filename'] = output_filename

    save_log(args.logfile, log)

    plot(outputs, results_dir)

    ros_init.shutdown()


if __name__ == '__main__':
    main()