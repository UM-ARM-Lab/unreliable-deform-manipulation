#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf

import link_bot_classifiers
import rospy
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import batch_tf_dataset
from link_bot_pycommon.pycommon import paths_to_json
from shape_completion_training.metric import AccuracyMetric
from shape_completion_training.model import filepath_tools
from shape_completion_training.model_runner import ModelRunner


def train_main(args, seed: int):
    ###############
    # Datasets
    ###############
    train_dataset = ClassifierDataset(args.dataset_dirs)
    val_dataset = ClassifierDataset(args.dataset_dirs)

    ###############
    # Model
    ###############
    model_hparams = json.load((args.model_hparams).open('r'))
    model_hparams['classifier_dataset_hparams'] = train_dataset.hparams
    model_hparams['batch_size'] = args.batch_size
    model_hparams['seed'] = seed
    model_hparams['datasets'] = paths_to_json(args.dataset_dirs)
    trial_path = args.checkpoint.absolute() if args.checkpoint is not None else None
    trials_directory = pathlib.Path('trials').absolute()
    group_name = args.log if trial_path is None else None
    trial_path, params = filepath_tools.create_or_load_trial(group_name=group_name,
                                                             params=model_hparams,
                                                             trial_path=trial_path,
                                                             trials_directory=trials_directory,
                                                             write_summary=False)
    model_class = link_bot_classifiers.get_model(model_hparams['model_class'])

    model = model_class(hparams=model_hparams, batch_size=args.batch_size, scenario=train_dataset.scenario)

    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         trial_path=trial_path,
                         key_metric=AccuracyMetric,
                         val_every_n_batches=100)

    # Dataset preprocessing
    train_tf_dataset = train_dataset.get_datasets(mode='train', take=args.take)
    val_tf_dataset = val_dataset.get_datasets(mode='val')

    # to mix up examples so each batch is diverse
    train_tf_dataset = train_tf_dataset.shuffle(buffer_size=2048, seed=seed, reshuffle_each_iteration=True)

    train_tf_dataset = batch_tf_dataset(train_tf_dataset, args.batch_size, drop_remainder=True)
    val_tf_dataset = batch_tf_dataset(val_tf_dataset, args.batch_size, drop_remainder=True)

    train_tf_dataset = train_tf_dataset.shuffle(buffer_size=512, seed=seed, reshuffle_each_iteration=True)  # to mix up batches

    train_tf_dataset = train_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_tf_dataset = val_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    runner.train(train_tf_dataset, val_tf_dataset, num_epochs=args.epochs)


def eval_main(args, seed: int):
    ###############
    # Model
    ###############
    _, params = filepath_tools.create_or_load_trial(trial_path=args.checkpoint.absolute(),
                                                    trials_directory=pathlib.Path('trials'))
    model = link_bot_classifiers.get_model(params['model_class'])

    ###############
    # Dataset
    ###############
    test_dataset = ClassifierDataset(args.dataset_dirs)
    test_tf_dataset = test_dataset.get_datasets(mode=args.mode)

    ###############
    # Evaluate
    ###############
    test_tf_dataset = test_tf_dataset.batch(args.batch_size, drop_remainder=True)

    net = model(hparams=params, batch_size=args.batch_size, scenario=test_dataset.scenario)
    runner = ModelRunner(model=net,
                         training=False,
                         trial_path=args.checkpoint.absolute(),
                         trials_directory=pathlib.Path('trials'),
                         write_summary=False,
                         key_metric=AccuracyMetric)
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
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--take', type=int)
    train_parser.add_argument('--debug', action='store_true')
    train_parser.add_argument('--epochs', type=int, default=10)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--log-scalars-every', type=int, help='loss/accuracy every this many steps/batches',
                              default=100)
    train_parser.add_argument('--validation-every', type=int, help='report validation every this many epochs',
                              default=1)
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.set_defaults(func=train_main)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    eval_parser.add_argument('--batch-size', type=int, default=32)
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

    rospy.init_node("train_test_classifier")

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args, seed)


if __name__ == '__main__':
    main()
