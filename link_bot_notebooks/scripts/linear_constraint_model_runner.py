#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys
import os
import numpy as np

from link_bot_notebooks import linear_constraint_model as m
from link_bot_notebooks import experiments_util


def load_sdf(filename):
    npz = np.load(filename)
    return npz['sdf'], npz['sdf_gradient'], npz['sdf_resolution']


def train(args):
    log_path = experiments_util.experiment_name(args.log)
    log_data = np.load(args.dataset)
    sdf, sdf_gradient, sdf_resolution = load_sdf(args.sdf)
    x = log_data[:, :, :]
    dt = x[0, 1, 0] - x[0, 0, 0]
    model = m.LinearConstraintModel(vars(args), sdf, sdf_gradient, sdf_resolution, x.shape[0], args.N, args.M, args.L,
                                    args.P, args.Q, dt, x.shape[1] - 1,
                                    seed=args.seed)

    goal = np.zeros((1, args.N))

    model.setup()

    model.train(x, goal, args.epochs, log_path)

    model.evaluate(x, goal)


def model_only(args):
    W = 10
    H = 20
    model = m.LinearConstraintModel(vars(args), np.random.randn(W, H).astype(np.float32),
                                    np.random.randn(W, H, 2).astype(np.float32), np.random.randn(2).astype(np.float32),
                                    250, args.N, args.M, args.L, args.P, args.Q, 0.1, 50)
    if args.log:
        model.init()
        log_path = experiments_util.experiment_name(args.log)
        full_log_path = os.path.join(os.getcwd(), "log_data", log_path)
        model.save(full_log_path)


def evaluate(args):
    goal = np.zeros((1, args.N))
    log_data = np.load(args.dataset)
    sdf, sdf_gradient, sdf_resolution = load_sdf(args.sdf)
    x = log_data[:, :, :]
    dt = x[0, 1, 0] - x[0, 0, 0]
    model = m.LinearConstraintModel(vars(args), sdf, sdf_gradient, sdf_resolution, x.shape[0], args.N, args.M, args.L,
                                    args.P, dt, x.shape[1] - 1)
    model.load()
    return model.evaluate(x, goal)


def main():
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("-M", help="dimensions in latent state o_d", type=int, default=2)
    parser.add_argument("-L", help="dimensions in control input", type=int, default=2)
    parser.add_argument("-P", help="dimensions in latent state o_k", type=int, default=2)
    parser.add_argument("-Q", help="dimensions in constraint checking output space", type=int, default=1)
    parser.add_argument("--debug", help="enable TF Debugger", action='store_true')

    subparsers = parser.add_subparsers()
    train_subparser = subparsers.add_parser("train")
    train_subparser.add_argument("dataset", help="dataset (txt file)")
    train_subparser.add_argument("sdf", help="sdf and gradient of the environment (npz file)")
    train_subparser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=500)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.add_argument("--print-period", "-p", type=int, default=200)
    train_subparser.add_argument("--save-period", type=int, default=400)
    train_subparser.add_argument("--seed", type=int, default=0)
    train_subparser.set_defaults(func=train)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.add_argument("dataset", help="dataset (txt file)")
    eval_subparser.add_argument("checkpoint", help="eval the *.ckpt name")
    eval_subparser.set_defaults(func=evaluate)

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