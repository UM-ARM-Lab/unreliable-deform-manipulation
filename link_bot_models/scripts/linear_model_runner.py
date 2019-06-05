#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys
import os
import numpy as np

from link_bot_models.src.link_bot_models import linear_tf_model as m
from link_bot_pycommon.src.link_bot_pycommon import experiments_util


def train(args):
    log_path = experiments_util.experiment_name(args.log)
    data = np.load(args.dataset)
    times = data['times']
    dt = times[0, 1, 0] - times[0, 0, 0]
    batch_size = data['states'].shape[0]
    n_steps = times.shape[1] - 1
    model = m.LinearTFModel(vars(args), batch_size, args.N, args.M, args.L, dt, n_steps, seed=args.seed)

    goal = np.zeros((1, args.N))

    model.setup()

    model.train(data, goal, args.epochs, log_path)

    model.evaluate(data, goal)


def model_only(args):
    model = m.LinearTFModel(vars(args), batch_size=250, N=args.N, M=args.M, L=args.L, n_steps=10, dt=0.1)
    if args.log:
        model.init()
        log_path = experiments_util.experiment_name(args.log)
        full_log_path = os.path.join(os.getcwd(), "log_data", log_path)
        model.save(full_log_path)


def evaluate(args):
    goal = np.zeros((1, args.N))
    data = np.load(args.dataset)
    times = data['times']
    dt = times[0, 1, 0] - times[0, 0, 0]
    batch_size = data['states'].shape[0]
    n_steps = times.shape[1] - 1
    model = m.LinearTFModel(vars(args), batch_size, args.N, args.M, args.L, dt, n_steps, seed=args.seed)
    model.load()
    return model.evaluate(data, goal)

def show(args):
    model = m.LinearTFModel(vars(args), 250, args.N, args.M, args.L, 0.1, 1)
    model.setup()
    print(model)



def main():
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("-M", help="dimensions in latent state", type=int, default=2)
    parser.add_argument("-L", help="dimensions in control input", type=int, default=2)
    parser.add_argument("--debug", help="enable TF Debugger", action='store_true')
    parser.add_argument("--seed", type=int, default=0)

    subparsers = parser.add_subparsers()
    train_subparser = subparsers.add_parser("train")
    train_subparser.add_argument("dataset", help="dataset (txt file)")
    train_subparser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=200)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.add_argument("--print-period", "-p", type=int, default=200)
    train_subparser.add_argument("--save-period", type=int, default=400)
    train_subparser.set_defaults(func=train)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.add_argument("dataset", help="dataset (txt file)")
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
