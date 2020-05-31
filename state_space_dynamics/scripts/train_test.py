#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf

import state_space_dynamics
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_pycommon.get_scenario import get_scenario
from moonshine.gpu_config import limit_gpu_mem
from shape_completion_training.model_runner import ModelRunner

limit_gpu_mem(3)


def train_main(args, seed: int):
    ###############
    # Datasets
    ###############
    train_dataset = DynamicsDataset(args.dataset_dirs)
    val_dataset = DynamicsDataset(args.dataset_dirs)

    ###############
    # Model
    ###############
    model_hparams = json.load((args.model_hparams).open('r'))
    model_hparams['dynamics_dataset_hparams'] = train_dataset.hparams
    model_hparams['batch_size'] = args.batch_size
    model_hparams['seed'] = seed
    model_class = state_space_dynamics.get_model(model_hparams['model_class'])
    scenario = get_scenario(train_dataset.hparams['scenario'])

    # Dataset preprocessing
    train_tf_dataset = train_dataset.get_datasets(mode='train', take=args.take)
    val_tf_dataset = val_dataset.get_datasets(mode='val', take=200)

    # to mix up examples so each batch is diverse
    train_tf_dataset = train_tf_dataset.shuffle(buffer_size=2048, seed=seed, reshuffle_each_iteration=True)

    train_tf_dataset = train_tf_dataset.batch(args.batch_size, drop_remainder=True)
    val_tf_dataset = val_tf_dataset.batch(args.batch_size, drop_remainder=True)

    train_tf_dataset = train_tf_dataset.shuffle(buffer_size=512, seed=seed, reshuffle_each_iteration=True)  # to mix up batches

    train_tf_dataset = train_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_tf_dataset = val_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = model_class(hparams=model_hparams, batch_size=args.batch_size, scenario=scenario)

    # Train
    trial_path = args.checkpoint.absolute() if args.checkpoint is not None else None
    groupname = args.log if trial_path is None else None
    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         trial_path=trial_path,
                         group_name=groupname,
                         trials_directory=pathlib.Path('trials'),
                         write_summary=False)
    runner.train(train_tf_dataset, val_tf_dataset, num_epochs=args.epochs)


def eval_main(args, seed: int):
    ###############
    # Model
    ###############
    model_hparams = json.load((args.checkpoint / 'params.json').open('r'))
    model = state_space_dynamics.get_model(model_hparams['model_class'])
    scenario = get_scenario(model_hparams['scenario'])
    net = model(hparams=model_hparams, batch_size=args.batch_size, scenario=scenario)

    ###############
    # Dataset
    ###############
    test_dataset = DynamicsDataset(args.dataset_dirs)
    test_tf_dataset = test_dataset.get_datasets(mode=args.mode)

    ###############
    # Evaluate
    ###############
    test_tf_dataset = test_tf_dataset.batch(args.batch_size, drop_remainder=True)

    runner = ModelRunner(model=model,
                         training=True,
                         params=model_hparams,
                         group_name=args.log,
                         trials_directory=pathlib.Path('trials'),
                         write_summary=False)
    runner.val_epoch(test_tf_dataset)


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

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args, seed)


if __name__ == '__main__':
    main()
