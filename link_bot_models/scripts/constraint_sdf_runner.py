#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from link_bot_models.constraint_sdf import ConstraintSDF
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import experiments_util

label_types = [LabelType.SDF]


def train(args):
    log_path = experiments_util.experiment_name(args.log)

    train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    validation_dataset = MultiEnvironmentDataset.load_dataset(args.validation_dataset)
    sdf_shape = train_dataset.sdf_shape

    model = ConstraintSDF(vars(args), sdf_shape, args.N)
    model.setup()
    model.train(train_dataset, validation_dataset, label_types, args.epochs, log_path)

    model.evaluate(validation_dataset, label_types)


def model_only(args):
    args_dict = vars(args)
    args_dict['random_init'] = False
    model = ConstraintSDF(args_dict, [10, 10], args.N)

    model.init()

    if args.log:
        log_path = experiments_util.experiment_name(args.log)
        experiments_util.make_log_dir(log_path)
        full_log_path = os.path.join(os.getcwd(), "log_data", log_path)
        model.save(full_log_path)

    print(model)


def evaluate(args):
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    sdf_shape = dataset.sdf_shape

    args_dict = vars(args)
    args_dict['random_init'] = False
    model = ConstraintSDF(args_dict, sdf_shape, args.N)
    model.setup()

    return model.evaluate(dataset, label_types)


def show(args):
    args_dict = vars(args)
    args_dict['random_init'] = False
    model = ConstraintSDF(args_dict, [10, 10], args.N)
    model.setup()
    print(model)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=220)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("--debug", help="enable TF Debugger", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("label_type", type=LabelType.from_string, choices=list(LabelType))

    subparsers = parser.add_subparsers()
    train_subparser = subparsers.add_parser("train")
    train_subparser.add_argument("train_dataset", help="dataset (json file)")
    train_subparser.add_argument("validation_dataset", help="dataset (json file)")
    train_subparser.add_argument("--batch-size", "-b", type=int, default=200)
    train_subparser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=250)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.add_argument("--log-period", type=int, default=250)
    train_subparser.add_argument("--print-period", type=int, default=250)
    train_subparser.add_argument("--val-period", type=int, default=10, help='run validation every so many epochs')
    train_subparser.add_argument("--save-period", type=int, default=20, help='save every so many epochs')
    train_subparser.add_argument("--random-init", action='store_true')
    train_subparser.set_defaults(func=train)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.add_argument("dataset", help="dataset (json file)")
    eval_subparser.add_argument("checkpoint", help="eval the *.ckpt name")
    eval_subparser.add_argument("--batch-size", "-b", type=int, default=100)
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
