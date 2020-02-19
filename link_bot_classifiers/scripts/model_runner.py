#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf
from colorama import Fore

import link_bot_classifiers
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import balance_by_augmentation
from link_bot_pycommon import experiments_util

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train(args, seed: int):
    if args.log:
        log_path = experiments_util.experiment_name(args.log)
    else:
        log_path = None

    ###############
    # Datasets
    ###############
    classifier_dataset_params = json.load(args.classifier_dataset_params.open('r'))
    train_dataset = ClassifierDataset(args.dataset_dirs, classifier_dataset_params)
    val_dataset = ClassifierDataset(args.dataset_dirs, classifier_dataset_params)

    train_tf_dataset = train_dataset.get_datasets(mode='train',
                                                  seed=seed)
    val_tf_dataset = val_dataset.get_datasets(mode='val',
                                              seed=seed)

    ###############
    # Model
    ###############
    model_hparams = json.load(args.model_hparams.open('r'))
    # FIXME: just garbage
    model_hparams['classifier_dataset_params'] = classifier_dataset_params
    model_hparams['classifier_dataset_hparams'] = train_dataset.hparams
    module = link_bot_classifiers.get_model_module(model_hparams['model_class'])

    # FIXME: this is a different net, as the one we are training but has the same hparams...
    #  technically that's fine but is bad design
    net = module.model(model_hparams, batch_size=args.batch_size)
    train_tf_dataset = net.post_process(train_tf_dataset)
    val_tf_dataset = net.post_process(val_tf_dataset)

    if classifier_dataset_params['balance']:
        # TODO: make this faster somehow
        print(Fore.GREEN + "balancing..." + Fore.RESET)
        train_tf_dataset = balance_by_augmentation(train_tf_dataset, key='label')
        val_tf_dataset = balance_by_augmentation(val_tf_dataset, key='label')
    
    try:
        ###############
        # Train
        ###############
        train_tmp = "/tmp/tf_train_{}".format(100000)
        val_tmp = "/tmp/tf_val_{}".format(100000)
        # train_tf_dataset = train_tf_dataset.cache(train_tmp).shuffle(buffer_size=1024, seed=seed).batch(args.batch_size, drop_remainder=True)
        # val_tf_dataset = val_tf_dataset.cache(val_tmp).batch(args.batch_size, drop_remainder=True)
        train_tf_dataset = train_tf_dataset.shuffle(buffer_size=1024, seed=seed).batch(args.batch_size, drop_remainder=True)
        val_tf_dataset = val_tf_dataset.batch(args.batch_size, drop_remainder=True)
        module.train(model_hparams, train_tf_dataset, val_tf_dataset, log_path, args)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "Interrupted." + Fore.RESET)
        pass


def eval(args, seed: int):
    ###############
    # Model
    ###############
    model_hparams = json.load((args.checkpoint / 'hparams.json').open('r'))
    module = link_bot_classifiers.get_model_module(model_hparams['model_class'])

    ###############
    # Dataset
    ###############
    classifier_dataset_params = model_hparams['labeling_hparams']
    test_dataset = ClassifierDataset(args.dataset_dirs, classifier_dataset_params)

    test_tf_dataset = test_dataset.get_datasets(mode=args.mode,
                                                seed=seed)

    try:
        ###############
        # Evaluate
        ###############
        module.eval(model_hparams, test_tf_dataset, args)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "Interrupted." + Fore.RESET)
        pass


def main():
    np.set_printoptions(linewidth=250, precision=4, suppress=True)
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('classifier_dataset_params', type=pathlib.Path)
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
    train_parser.set_defaults(func=train)
    train_parser.add_argument('--seed', type=int, default=None)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--mode', type=str, choices=['test', 'val', 'train'], default='test')
    eval_parser.add_argument('--batch-size', type=int, default=32)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.set_defaults(func=eval)
    eval_parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    if args.seed is None:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    print("Using seed {}".format(seed))
    np.random.seed(seed)
    tf.random.set_random_seed(seed)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args, seed)


if __name__ == '__main__':
    main()
