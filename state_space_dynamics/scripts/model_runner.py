#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf
from colorama import Fore

import state_space_dynamics
from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset
from link_bot_pycommon import experiments_util

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)


def train(args):
    if args.log:
        log_path = experiments_util.experiment_name(args.log)
    else:
        log_path = None

    # Model parameters
    model_hparams = json.load(open(args.model_hparams, 'r'))

    # Datasets
    train_dataset = LinkBotStateSpaceDataset(args.dataset_dir)
    train_tf_dataset = train_dataset.get_dataset(mode='train',
                                                 shuffle=True,
                                                 seed=args.seed,
                                                 sequence_length=model_hparams['sequence_length'],
                                                 batch_size=args.batch_size)
    val_dataset = LinkBotStateSpaceDataset(args.dataset_dir)
    val_tf_dataset = val_dataset.get_dataset(mode='val',
                                             shuffle=True,
                                             seed=args.seed,
                                             sequence_length=model_hparams['sequence_length'],
                                             batch_size=args.batch_size)

    # Copy parameters of the dataset into the model
    model_hparams['dynamics_dataset_hparams'] = train_dataset.hparams
    module = state_space_dynamics.get_model_module(model_hparams['model_class'])

    try:
        ###############
        # Train
        ###############
        module.train(model_hparams, train_tf_dataset, val_tf_dataset, log_path, args)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "Interrupted." + Fore.RESET)
        pass


def eval(args):
    ###############
    # Dataset
    ###############
    test_dataset = LinkBotStateSpaceDataset(args.dataset_dir)
    test_tf_dataset = test_dataset.get_dataset(mode='test',
                                               shuffle=False,
                                               seed=args.seed,
                                               batch_size=args.batch_size)

    ###############
    # Model
    ###############
    model_hparams_file = args.checkpoint / 'hparams.json'
    model_hparams = json.load(open(model_hparams_file, 'r'))
    model_hparams['dt'] = test_dataset.hparams['dt']

    module = state_space_dynamics.get_model_module(model_hparams['model_class'])

    try:
        ###############
        # Evaluate
        ###############
        module.eval(model_hparams, test_tf_dataset, args)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "Interrupted." + Fore.RESET)
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dir', type=pathlib.Path)
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--summary-freq', type=int, default=5)
    train_parser.add_argument('--save-freq', type=int, default=10)
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--validation-every', type=int, help='report validation every this many epochs', default=4)
    train_parser.add_argument('--debug', action='store_true')
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dir', type=pathlib.Path)
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--batch-size', type=int, default=32)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.set_defaults(func=eval)

    args = parser.parse_args()

    if args.seed is None:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    print("Random seed: {}".format(seed))
    np.random.seed(seed)
    tf.random.set_random_seed(seed)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
