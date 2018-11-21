#!/usr/bin/env python

import argparse
import numpy as np

from link_bot_notebooks import notebook_finder
from link_bot_notebooks import toy_problem_optimization_common as tpo
from link_bot_notebooks.nn_model import NNModel
from link_bot_notebooks.linear_tf_model import LinearTFModel

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
    n, x, y = tpo.load_train_test(args.dataset, N=6, M=2, L=2, g=goal, extract_func=tpo.two_link_pos_vel_extractor)
    # model = NNModel(args, N=6, M=2, L=2, dims=DIMENSIONS)
    model = LinearTFModel(vars(args), N=6, M=2, L=2)
    model.train(x, y, args.epochs)


def model_only(args):
    # model = NNModel(args, N=6, M=2, L=2, dims=DIMENSIONS)
    model = LinearTFModel(vars(args), N=6, M=2, L=2)
    if args.log:
        model.init()
        model.save()

def evaluate(args):
    goal = np.array([4, 0, 5, 0, 6, 0])
    n, x, y = tpo.load_train_test(args.dataset, N=6, M=2, L=2, g=goal, extract_func=tpo.two_link_pos_vel_extractor)
    # model = NNModel(args, N=6, M=2, L=2, dims=DIMENSIONS)
    model = LinearTFModel(vars(args), N=6, M=2, L=2)
    model.load()
    model.evaluate(x, y)


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true')

    subparsers = parser.add_subparsers()
    train_subparser = subparsers.add_parser("train")
    train_subparser.add_argument("dataset", help="dataset (txt file)")
    train_subparser.add_argument("--log", "-l", action="store_true", help="save/log the graph and summaries")
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=100)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.add_argument("--batch_size", "-b", type=int, default=-1)
    train_subparser.set_defaults(func=train)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.add_argument("dataset", help="dataset (txt file)")
    eval_subparser.add_argument("checkpoint", help="eval the *.ckpt name")
    eval_subparser.set_defaults(func=evaluate)

    model_only_subparser = subparsers.add_parser("model_only")
    model_only_subparser.add_argument("--log", "-l", action="store_true", help="save/log the graph and summaries")
    model_only_subparser.set_defaults(func=model_only)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)
