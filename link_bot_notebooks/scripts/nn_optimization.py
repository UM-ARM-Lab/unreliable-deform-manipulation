#!/usr/bin/env python

import argparse
import numpy as np

from link_bot_notebooks.nn_model import NNModel, load_and_construct_training_data

DIMENSIONS = {
    # Reduction
    'F1_n': 28,
    'F2_n': 16,
    # Dynamics
    'T1_n': 16,
    'T2_n': 16
}


def train(args):
    goal = np.array([4, 0, 5, 0, 6, 0])
    n, x, y = load_and_construct_training_data(args.dataset, 6, 2, 2, goal)
    model = NNModel(args, N=6, M=2, L=2, dims=DIMENSIONS)
    model.train(x, y, args.epochs)


def model_only(args):
    model = NNModel(args, N=6, M=2, L=2, dims=DIMENSIONS)
    if args.log:
        model.init()
        model.save()


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true')

    subparsers = parser.add_subparsers()
    train_subparser = subparsers.add_parser("train")
    train_subparser.add_argument("dataset", help="dataset (txt file)")
    train_subparser.add_argument("--log", "-l", action="store_true", help="save/log the graph and summaries")
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=100)
    train_subparser.set_defaults(func=train)

    model_only_subparser = subparsers.add_parser("model_only")
    model_only_subparser.add_argument("--log", "-l", action="store_true", help="save/log the graph and summaries")
    model_only_subparser.set_defaults(func=model_only)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)
