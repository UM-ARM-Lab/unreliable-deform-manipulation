#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf
from colorama import Fore

import link_bot_classifiers
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_pycommon import experiments_util

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)

tf.logging.set_verbosity(tf.logging.ERROR)


def train(args):
    if args.log:
        log_path = experiments_util.experiment_name(args.log)
    else:
        log_path = None

    model_hparams = json.load(args.model_hparams.open('r'))

    ###############
    # Datasets
    ###############
    train_classifier_dataset = ClassifierDataset(args.input_dir)
    train_dataset = train_classifier_dataset.get_dataset(mode='train',
                                                         shuffle=True,
                                                         num_epochs=1,
                                                         seed=args.seed,
                                                         batch_size=args.batch_size,
                                                         balance_key=args.balance_key)
    val_classifier_dataset = ClassifierDataset(args.input_dir)
    val_dataset = val_classifier_dataset.get_dataset(mode='val',
                                                     shuffle=True,
                                                     num_epochs=1,
                                                     seed=args.seed,
                                                     batch_size=args.batch_size,
                                                     balance_key=args.balance_key)

    ###############
    # Model
    ###############
    module = link_bot_classifiers.get_model_module(model_hparams['model_class'])

    try:
        ###############
        # Train
        ###############
        module.train(model_hparams, train_dataset, val_dataset, log_path, args)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "Interrupted." + Fore.RESET)
        pass


def eval(args):
    if args.dataset_hparams_dict:
        dataset_hparams_dict = json.load(args.dataset_hparams_dict.open('r'))
    else:
        dataset_hparams_dict = json.load((args.input_dir / 'hparams.json').open('r'))

    model_hparams_file = args.checkpoint / 'hparams.json'
    model_hparams = json.load(model_hparams_file.open('r'))
    dataset_hparams_dict['local_env_shape'] = model_hparams['local_env_shape']

    ###############
    # Dataset
    ###############
    test_classifier_dataset = ClassifierDataset(args.input_dir)
    test_dataset = test_classifier_dataset.get_dataset(mode=args.mode,
                                                       shuffle=False,
                                                       num_epochs=1,
                                                       seed=args.seed,
                                                       batch_size=args.batch_size,
                                                       balance_key=args.balance_key)

    ###############
    # Model
    ###############
    module = link_bot_classifiers.get_model_module(model_hparams['model_class'])

    try:
        ###############
        # Evaluate
        ###############
        module.eval(model_hparams, test_dataset, args)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "Interrupted." + Fore.RESET)
        pass


def main():
    np.set_printoptions(linewidth=250)
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('input_dir', type=pathlib.Path)
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--dataset-hparams-dict', type=pathlib.Path)
    train_parser.add_argument('--dataset-hparams', type=str)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--summary-freq', type=int, default=1)
    train_parser.add_argument('--save-freq', type=int, default=1)
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--log-grad-every', type=int, help='gradients hists every this many steps/batches', default=1000)
    train_parser.add_argument('--log-scalars-every', type=int, help='loss/accuracy every this many steps/batches', default=500)
    train_parser.add_argument('--validation-every', type=int, help='report validation every this many epochs', default=2000)
    train_parser.add_argument('--balance-key', help='use this key in the y of dataset to balance classes')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('--seed', type=int, default=None)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('input_dir', type=pathlib.Path)
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--dataset-hparams-dict', type=pathlib.Path)
    eval_parser.add_argument('--dataset-hparams', type=str)
    eval_parser.add_argument('--mode', type=str, choices=['test', 'val', 'train'], default='test')
    eval_parser.add_argument('--batch-size', type=int, default=32)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.add_argument('--balance-key', help='use this key in the y of dataset to balance classes')
    eval_parser.set_defaults(func=eval)
    eval_parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    if args.seed is None:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    np.random.seed(seed)
    tf.random.set_random_seed(seed)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
