#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf
from colorama import Fore

import state_space_dynamics
from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset
from moonshine import experiments_util

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train(args, seed):
    if args.log:
        log_path = experiments_util.experiment_name(args.log)
    else:
        log_path = None

    # Model parameters
    model_hparams = json.load(open(args.model_hparams, 'r'))

    # Datasets
    train_dataset = LinkBotStateSpaceDataset(args.dataset_dirs)
    train_tf_dataset = train_dataset.get_datasets(mode='train', sequence_length=model_hparams['sequence_length'])
    val_dataset = LinkBotStateSpaceDataset(args.dataset_dirs)
    val_tf_dataset = val_dataset.get_datasets(mode='val', sequence_length=model_hparams['sequence_length'])
    train_tf_dataset = train_tf_dataset.shuffle(seed=args.seed, buffer_size=1024).batch(args.batch_size, drop_remainder=True)
    val_tf_dataset = val_tf_dataset.batch(args.batch_size, drop_remainder=True)

    # Copy parameters of the dataset into the model
    model_hparams['dynamics_dataset_hparams'] = train_dataset.hparams
    module = state_space_dynamics.get_model_module(model_hparams['model_class'])

    try:
        ###############
        # Train
        ###############
        module.train(model_hparams, train_tf_dataset, val_tf_dataset, log_path, args, seed)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "Interrupted." + Fore.RESET)
        pass


def eval(args, seed):
    ###############
    # Dataset
    ###############
    test_dataset = LinkBotStateSpaceDataset(args.dataset_dirs)
    test_tf_dataset = test_dataset.get_datasets(mode=args.mode,
                                                sequence_length=args.sequence_length,
                                                )

    test_tf_dataset = test_tf_dataset.batch(args.batch_size, drop_remainder=True)

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
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--summary-freq', type=int, default=5)
    train_parser.add_argument('--save-freq', type=int, default=10)
    train_parser.add_argument('--epochs', type=int, default=300)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--validation-every', type=int, help='report validation every this many epochs', default=4)
    train_parser.add_argument('--debug', action='store_true')
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--sequence-length', type=int, default=10)
    eval_parser.add_argument('--batch-size', type=int, default=32)
    eval_parser.add_argument('--mode', type=str, choices=['test', 'val', 'train'], default='test')
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
        args.func(args, seed)


if __name__ == '__main__':
    main()
