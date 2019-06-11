#!/usr/bin/env python
from __future__ import print_function

import argparse
from colorama import Fore
import sys
import os
import tensorflow as tf
import numpy as np

from link_bot_models.constraint_model import ConstraintModelType, ConstraintModel
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import link_bot_pycommon, experiments_util


def train(args):
    log_path = experiments_util.experiment_name(args.log)
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    # sdf_data = link_bot_pycommon.load_sdf_data(args.sdf)
    # data = np.load(args.dataset)
    model = ConstraintModel(vars(args), dataset, args.N)

    model.setup()

    split_data = model.split_data(dataset)
    train_observations, train_k, validation_observations, validation_k = split_data
    model.train(*split_data, args.epochs, log_path)

    print(Fore.GREEN + "\nTrain Evaluation" + Fore.RESET)
    model.evaluate(train_observations, train_k)
    print(Fore.GREEN + "\nTest Evaluation" + Fore.RESET)
    model.evaluate(validation_observations, validation_k)


def model_only(args):
    W = 10
    H = 20
    fake_sdf = np.random.randn(W, H).astype(np.float32)
    fake_sdf_grad = np.random.randn(W, H, 2).astype(np.float32)
    fake_sdf_res = np.random.randn(2).astype(np.float32)
    fake_sdf_origin = np.random.randn(2).astype(np.float32)
    sdf_data = link_bot_pycommon.SDF(fake_sdf, fake_sdf_grad, fake_sdf_res, fake_sdf_origin, None, None)
    args_dict = vars(args)
    args_dict['random_init'] = False
    model = ConstraintModel(args_dict, sdf_data, args.N)

    model.init()

    if args.log:
        log_path = experiments_util.experiment_name(args.log)
        experiments_util.make_log_dir(log_path)
        full_log_path = os.path.join(os.getcwd(), "log_data", log_path)
        model.save(full_log_path)

    print(model)


def evaluate(args):
    sdf_data = link_bot_pycommon.load_sdf_data(args.sdf)
    data = np.load(args.dataset)
    args_dict = vars(args)
    args_dict['random_init'] = False
    model = ConstraintModel(args_dict, sdf_data, args.N)
    model.setup()

    # take all the data as validation data
    split_data = model.split_data(data, fraction_validation=1.00)
    train_observations, train_k, validation_observations, validation_k = split_data

    return model.evaluate(validation_observations, validation_k)


def show(args):
    W = 10
    H = 20
    fake_sdf = np.random.randn(W, H).astype(np.float32)
    fake_sdf_grad = np.random.randn(W, H, 2).astype(np.float32)
    fake_sdf_res = np.random.randn(2).astype(np.float32)
    fake_sdf_origin = np.random.randn(2).astype(np.float32)
    sdf_data = link_bot_pycommon.SDF(fake_sdf, fake_sdf_grad, fake_sdf_res, fake_sdf_origin, None, None)
    args_dict = vars(args)
    args_dict['random_init'] = False
    model = ConstraintModel(args_dict, sdf_data, args.N)
    model.setup()
    print(model)


def main():
    np.set_printoptions(precision=6, suppress=True)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("--debug", help="enable TF Debugger", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("model_type", type=ConstraintModelType.from_string, choices=list(ConstraintModelType))

    subparsers = parser.add_subparsers()
    train_subparser = subparsers.add_parser("train")
    train_subparser.add_argument("dataset", help="dataset (json file)")
    # train_subparser.add_argument("dataset", help="dataset")
    # train_subparser.add_argument("sdf", help="sdf and gradient of the environment (npz file)")
    train_subparser.add_argument("--batch-size", "-b", type=int, default=128)
    train_subparser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=50000)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.add_argument("--print-period", "-p", type=int, default=100)
    train_subparser.add_argument("--save-period", type=int, default=1000)
    train_subparser.add_argument("--random-init", action='store_true')
    train_subparser.set_defaults(func=train)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.add_argument("dataset", help="dataset (json file)")
    # eval_subparser.add_argument("sdf", help="sdf and gradient of the environment (npz file)")
    eval_subparser.add_argument("checkpoint", help="eval the *.ckpt name")
    eval_subparser.set_defaults(func=evaluate)

    show_subparser = subparsers.add_parser("show")
    show_subparser.add_argument("checkpoint", help="restart from this *.ckpt name")
    show_subparser.set_defaults(func=show)

    model_only_subparser = subparsers.add_parser("model_only")
    model_only_subparser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    model_only_subparser.set_defaults(func=model_only)

    args = parser.parse_args()
    commandline = ' '.join(sys.argv)
    args.commandline = commandline

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
