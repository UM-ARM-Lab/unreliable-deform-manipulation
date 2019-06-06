#!/usr/bin/env python
from __future__ import print_function

import argparse
import sys
import os
import numpy as np

from link_bot_models.src.link_bot_models import linear_constraint_model as m
from link_bot_pycommon.src.link_bot_pycommon import link_bot_pycommon as tpo, experiments_util


def train(args):
    log_path = experiments_util.experiment_name(args.log)
    sdf, sdf_gradient, sdf_resolution = tpo.load_sdf(args.sdf)
    data = np.load(args.dataset)
    if 'times' in data:
        times = data['times']
        dt = times[0, 1, 0] - times[0, 0, 0]
    else:
        dt = 0.1
    batch_size, n_steps, _ = data['actions'].shape
    model = m.LinearConstraintModel(vars(args), sdf, sdf_gradient, sdf_resolution, batch_size, args.N, args.M, args.L,
                                    args.P, args.Q, dt, n_steps)

    goal = np.zeros((1, args.N))

    model.setup()

    model.train(data, goal, args.epochs, log_path)

    model.evaluate(data, goal)


def model_only(args):
    W = 10
    H = 20
    fake_sdf = np.random.randn(W, H).astype(np.float32)
    fake_sdf_grad = np.random.randn(W, H, 2).astype(np.float32)
    fake_sdf_res = np.random.randn(2).astype(np.float32)
    model = m.LinearConstraintModel(vars(args), fake_sdf, fake_sdf_grad, fake_sdf_res, 250, args.N, args.M, args.L,
                                    args.P, args.Q, 0.1, 50)

    model.init()

    if args.log:
        log_path = experiments_util.experiment_name(args.log)
        experiments_util.make_log_dir(log_path)
        full_log_path = os.path.join(os.getcwd(), "log_data", log_path)
        model.save(full_log_path)

    print(model)


def evaluate(args):
    goal = np.zeros((1, args.N))
    sdf, sdf_gradient, sdf_resolution = tpo.load_sdf(args.sdf)
    data = np.load(args.dataset)
    if 'times' in data:
        times = data['times']
        dt = times[0, 1, 0] - times[0, 0, 0]
    else:
        dt = 0.1
    batch_size, n_steps, _ = data['actions'].shape
    print(n_steps)
    args_dict = vars(args)
    args_dict['random_init'] = False
    model = m.LinearConstraintModel(args_dict, sdf, sdf_gradient, sdf_resolution, batch_size, args.N, args.M, args.L,
                                    args.P, args.Q, dt, n_steps)
    model.setup()
    return model.evaluate(data, goal)


def show(args):
    W = 10
    H = 20
    fake_sdf = np.random.randn(W, H).astype(np.float32)
    fake_sdf_grad = np.random.randn(W, H, 2).astype(np.float32)
    fake_sdf_res = np.random.randn(2).astype(np.float32)
    args_dict = vars(args)
    args_dict['random_init'] = False
    model = m.LinearConstraintModel(args_dict, fake_sdf, fake_sdf_grad, fake_sdf_res, 250, args.N, args.M, args.L,
                                    args.P, args.Q, 0.1, 1)
    model.setup()
    print(model)


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
    parser.add_argument("--seed", type=int, default=0)

    subparsers = parser.add_subparsers()
    train_subparser = subparsers.add_parser("train")
    train_subparser.add_argument("dataset", help="dataset (txt file)")
    train_subparser.add_argument("sdf", help="sdf and gradient of the environment (npz file)")
    train_subparser.add_argument("--log", "-l", nargs='?', help="save/log the graph and summaries", const="")
    train_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=500)
    train_subparser.add_argument("--checkpoint", "-c", help="restart from this *.ckpt name")
    train_subparser.add_argument("--print-period", "-p", type=int, default=100)
    train_subparser.add_argument("--save-period", type=int, default=400)
    train_subparser.add_argument("--random-init", action='store_true')
    train_subparser.set_defaults(func=train)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.add_argument("dataset", help="dataset (txt file)")
    eval_subparser.add_argument("sdf", help="sdf and gradient of the environment (npz file)")
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