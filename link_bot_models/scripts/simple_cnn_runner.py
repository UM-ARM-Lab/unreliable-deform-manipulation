#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

from link_bot_models.simple_cnn_model import SimpleCNNModel
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import experiments_util

label_types = [LabelType.SDF]


def train(args):
    log_path = experiments_util.experiment_name(args.log)

    train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    validation_dataset = MultiEnvironmentDataset.load_dataset(args.validation_dataset)
    sdf_shape = train_dataset.sdf_shape

    if args.checkpoint:
        model = SimpleCNNModel.load(vars(args), sdf_shape, args.N)
    else:
        model = SimpleCNNModel(vars(args), sdf_shape, args.N)

    model.train(train_dataset, validation_dataset, label_types, args.epochs, log_path)


def evaluate(args):
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    sdf_shape = dataset.sdf_shape

    model = SimpleCNNModel.load(vars(args), sdf_shape, args.N)

    return model.evaluate(dataset, label_types)


def main():
    np.set_printoptions(precision=6, suppress=True)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("--debug", help="enable TF Debugger", action='store_true')
    parser.add_argument("--seed", type=int, default=0)

    subparsers = parser.add_subparsers()
    train_subparser = subparsers.add_parser("train")
    train_subparser.add_argument("train_dataset", help="dataset (json file)")
    train_subparser.add_argument("validation_dataset", help="dataset (json file)")
    train_subparser.add_argument("--batch-size", "-b", type=int, default=100)
    train_subparser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=250)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.add_argument("--plot", action="store_true")
    train_subparser.set_defaults(func=train)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.add_argument("dataset", help="dataset (json file)")
    eval_subparser.add_argument("checkpoint", help="eval the *.ckpt name")
    eval_subparser.add_argument("--batch-size", "-b", type=int, default=100)
    eval_subparser.set_defaults(func=evaluate)

    args = parser.parse_args()
    commandline = ' '.join(sys.argv)
    args.commandline = commandline

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
