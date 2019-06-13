#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from colorama import Fore

from link_bot_models import multi_environment_datasets
from link_bot_models.constraint_raster_cnn import ConstraintRasterCNN
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset, LabelType
from link_bot_pycommon import experiments_util


def train(args):
    log_path = experiments_util.experiment_name(args.log)
    train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    validation_dataset = MultiEnvironmentDataset.load_dataset(args.validation_dataset)
    sdf_shape = train_dataset.sdf_shape
    model = ConstraintRasterCNN(vars(args), sdf_shape, args.N)

    # convert data into two "lists" of numpy arrays
    train_inputs, train_labels = multi_environment_datasets.make_inputs_and_labels(train_dataset.environments)

    validation_inputs, validation_labels = multi_environment_datasets.make_inputs_and_labels(validation_dataset.environments)

    model.train(train_inputs, train_labels, validation_inputs, validation_labels, args.epochs, log_path)

    print(Fore.GREEN + "\nTrain Evaluation" + Fore.RESET)
    model.evaluate(train_inputs, train_labels)
    print(Fore.GREEN + "\nValidation Evaluation" + Fore.RESET)
    model.evaluate(validation_inputs, validation_labels)


def evaluate(args):
    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    sdf_shape = dataset.sdf_shape

    args_dict = vars(args)
    model = ConstraintRasterCNN(args_dict, sdf_shape, args.N)
    model.load()

    # take all the data as validation data
    validation_inputs, validation_labels = multi_environment_datasets.make_inputs_and_labels(dataset.environments)

    return model.evaluate(validation_inputs, validation_labels)


def main():
    np.set_printoptions(precision=6, suppress=True)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("--debug", help="enable TF Debugger", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("label_type", type=LabelType.from_string, choices=list(LabelType))

    subparsers = parser.add_subparsers()
    train_subparser = subparsers.add_parser("train")
    train_subparser.add_argument("train_dataset", help="dataset (json file)")
    train_subparser.add_argument("validation_dataset", help="dataset (json file)")
    train_subparser.add_argument("--batch-size", "-b", type=int, default=128)
    train_subparser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=125)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.set_defaults(func=train)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.add_argument("dataset", help="dataset (json file)")
    eval_subparser.add_argument("checkpoint", help="eval the *.ckpt name")
    eval_subparser.set_defaults(func=evaluate)

    args = parser.parse_args()
    commandline = ' '.join(sys.argv)
    args.commandline = commandline

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
