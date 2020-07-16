#!/usr/bin/env python
import rospy
import argparse
from colorama import Fore
import json
import pathlib

import numpy as np
import tensorflow as tf

import link_bot_classifiers
from link_bot_data.recovery_dataset import RecoveryDataset
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import paths_to_json
from shape_completion_training.metric import LossMetric
from link_bot_classifiers.nn_recovery_policy import NNRecoveryModel
from link_bot_data.link_bot_dataset_utils import balance, batch_tf_dataset
from moonshine.gpu_config import limit_gpu_mem
from shape_completion_training.model import filepath_tools
from shape_completion_training.model_runner import ModelRunner

limit_gpu_mem(6)


def train_main(args, seed: int):
    ###############
    # Datasets
    ###############
    train_dataset = RecoveryDataset(args.dataset_dirs)
    val_dataset = RecoveryDataset(args.dataset_dirs)

    ###############
    # Model
    ###############
    model_hparams = json.load((args.model_hparams).open('r'))
    model_hparams['recovery_dataset_hparams'] = train_dataset.hparams
    model_hparams['batch_size'] = args.batch_size
    model_hparams['seed'] = seed
    model_hparams['datasets'] = paths_to_json(args.dataset_dirs)
    scenario = get_scenario(model_hparams['scenario'])

    # Dataset preprocessing
    train_tf_dataset = train_dataset.get_datasets(mode='train', take=args.take)
    val_tf_dataset = val_dataset.get_datasets(mode='val')

    train_tf_dataset = batch_tf_dataset(train_tf_dataset, args.batch_size, drop_remainder=True)
    val_tf_dataset = batch_tf_dataset(val_tf_dataset, args.batch_size, drop_remainder=True)

    # FIXME: need to re-balance the dataset, but since it's not binary that's actually not easy to do
    # and in my experince just having class weights on the loss doesn't work very well,
    # so we should re-sample elements and write that out as a new dataset? maybe?

    train_tf_dataset = train_tf_dataset.shuffle(buffer_size=512, seed=seed)

    train_tf_dataset = train_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_tf_dataset = val_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = NNRecoveryModel(hparams=model_hparams, scenario=scenario, batch_size=args.batch_size)

    ############
    # Initialize weights from classifier model by "restoring" from checkpoint
    ############
    if not args.checkpoint:
        classifier_model = tf.train.Checkpoint(conv_layers=model.conv_layers, dense_layers=model.dense_layers)
        classifier_root = tf.train.Checkpoint(model=classifier_model)
        classifier_checkpoint_manager = tf.train.CheckpointManager(
            classifier_root, args.classifier_checkpoint.as_posix(), max_to_keep=1)

        status = classifier_root.restore(classifier_checkpoint_manager.latest_checkpoint)
        status.expect_partial()
        status.assert_existing_objects_matched()
        assert classifier_checkpoint_manager.latest_checkpoint is not None
        print(Fore.MAGENTA + "Restored {}".format(classifier_checkpoint_manager.latest_checkpoint) + Fore.RESET)
    ############

    trial_path = None
    checkpoint_name = None
    if args.checkpoint:
        trial_path = args.checkpoint.parent.absolute()
        checkpoint_name = args.checkpoint.name
    trials_directory = pathlib.Path('recovery_trials').absolute()
    group_name = args.log if trial_path is None else None
    trial_path, _ = filepath_tools.create_or_load_trial(group_name=group_name,
                                                        params=model_hparams,
                                                        trial_path=trial_path,
                                                        trials_directory=trials_directory,
                                                        write_summary=False)
    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         trial_path=trial_path,
                         restore_from_name=checkpoint_name,
                         batch_metadata=train_dataset.batch_metadata)

    # Train
    runner.train(train_tf_dataset, val_tf_dataset, num_epochs=args.epochs)


def eval_main(args, seed: int):
    ###############
    # Model
    ###############
    _, params = filepath_tools.create_or_load_trial(trial_path=args.checkpoint.absolute(),
                                                    trials_directory=pathlib.Path('trials'))
    model_class = link_bot_classifiers.get_model(params['model_class'])
    scenario = get_scenario(params['scenario'])
    net = model_class(hparams=params, scenario=scenario)

    ###############
    # Dataset
    ###############
    test_dataset = RecoveryDataset(args.dataset_dirs)
    test_tf_dataset = test_dataset.get_datasets(mode=args.mode)

    ###############
    # Evaluate
    ###############
    test_tf_dataset = test_tf_dataset.batch(args.batch_size, drop_remainder=True)

    runner = ModelRunner(model=net,
                         training=False,
                         trial_path=args.checkpoint.absolute(),
                         trials_directory=pathlib.Path('trials'),
                         write_summary=False)
    validation_metrics = runner.val_epoch(test_tf_dataset)
    for name, value in validation_metrics.items():
        print(f"{name}: {value:.3f}")


def main():
    np.set_printoptions(linewidth=250, precision=4, suppress=True, threshold=10000)
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('classifier_checkpoint', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--take', type=int)
    train_parser.add_argument('--epochs', type=int, default=10)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--log-scalars-every', type=int, help='loss every this many steps/batches',
                              default=100)
    train_parser.add_argument('--validation-every', type=int, help='report validation every this many epochs',
                              default=1)
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.set_defaults(func=train_main)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    eval_parser.add_argument('--batch-size', type=int, default=64)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.add_argument('--seed', type=int, default=None)
    eval_parser.set_defaults(func=eval_main)

    args = parser.parse_args()

    if args.seed is None:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    print("Using seed {}".format(seed))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    rospy.init_node("train_test_recovery")

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args, seed)


if __name__ == '__main__':
    main()
