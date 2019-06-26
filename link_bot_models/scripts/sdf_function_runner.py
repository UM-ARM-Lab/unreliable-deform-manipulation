#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

from link_bot_models import base_model
from link_bot_models.label_types import LabelType
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_models.sdf_function_model import SDFFunctionModel
from link_bot_pycommon import experiments_util

sdf_function_label_types = [LabelType.SDF]


def train(args):
    log_path = experiments_util.experiment_name(args.log)

    train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    validation_dataset = MultiEnvironmentDataset.load_dataset(args.validation_dataset)
    sdf_shape = train_dataset.sdf_shape

    if args.checkpoint:
        model = SDFFunctionModel.load(vars(args), sdf_shape, args.N)
    else:
        model = SDFFunctionModel(vars(args), sdf_shape, args.N)

    model.train(train_dataset, validation_dataset, sdf_function_label_types, args.epochs, log_path)
    model.evaluate(validation_dataset, sdf_function_label_types)


def evaluate(args):
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    sdf_shape = dataset.sdf_shape

    model = SDFFunctionModel.load(vars(args), sdf_shape, args.N)

    return model.evaluate(dataset, sdf_function_label_types)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=220)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser, train_subparser, eval_subparser, show_subparser = base_model.base_parser()

    train_subparser.add_argument("--sigmoid-scale", "-s", type=float, default=100)
    train_subparser.set_defaults(func=train)

    eval_subparser.set_defaults(func=evaluate)
    # FIXME: make hyper parameters loaded from metadata
    eval_subparser.add_argument("--sigmoid-scale", "-s", type=float, default=100)

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
