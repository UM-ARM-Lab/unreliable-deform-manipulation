#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

from link_bot_models.distance_function_model import DistanceFunctionModel
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_pycommon import experiments_util

label_types = [LabelType.Overstretching]


def train(args):
    log_path = experiments_util.experiment_name(args.log)

    train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    validation_dataset = MultiEnvironmentDataset.load_dataset(args.validation_dataset)

    if args.checkpoint:
        model = DistanceFunctionModel.load(vars(args), args.N)
    else:
        model = DistanceFunctionModel(vars(args), args.N)

    model.train(train_dataset, validation_dataset, label_types, args.epochs, log_path)


def evaluate(args):
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)

    model = DistanceFunctionModel.load(vars(args), args.N)

    # weights = model.keras_model.get_weights()
    # conv_kernel = np.squeeze(weights[0])
    # conv_bias = np.squeeze(weights[1])
    # print(conv_kernel)
    # print(conv_bias)
    # print(conv_kernel[0, 2] + conv_kernel[2, 0])
    # print(conv_kernel[0, 1] + conv_kernel[1, 0] + conv_kernel[1, 2] + conv_kernel[2, 1])

    # x = dataset.environments[0].rope_data['rope_configurations'][:1]
    # d = model.distance_matrix_model.predict(x)
    # print(np.squeeze(d))

    return model.evaluate(dataset, label_types)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=220)
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
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=50)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.add_argument("--random-init", action='store_true')
    train_subparser.add_argument("--plot", action='store_true')
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
