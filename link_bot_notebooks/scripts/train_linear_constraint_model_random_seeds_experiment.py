#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from link_bot_notebooks import experiments_util
from link_bot_notebooks import linear_constraint_model as m
from link_bot_notebooks import toy_problem_optimization_common as tpoc


def train(args):
    sdf, sdf_gradient, sdf_resolution = tpoc.load_sdf(args.sdf)
    data = np.load(args.dataset)
    times = data['times']
    dt = times[0, 1, 0] - times[0, 0, 0]
    batch_size = data['states'].shape[0]
    n_steps = times.shape[1] - 1
    args = vars(args)
    args['log'] = None
    args['checkpoint'] = None
    args['debug'] = False
    args['seed'] = 0
    N = 6
    M = 2
    L = 2
    P = 2
    Q = 1

    goal = np.zeros((1, N))
    arrays_to_save = {}
    name = experiments_util.experiment_name("attempt_train")
    outfile = os.path.join('test_data', name + ".npz")
    print("Saving results to: ", outfile)

    for i in range(args['num-training-attempts']):
        # initialize the weights
        tf.reset_default_graph()
        args['seed'] = i
        model = m.LinearConstraintModel(args, sdf, sdf_gradient, sdf_resolution, batch_size, N, M, L, P, Q, dt, n_steps)
        model.setup()

        # train
        interrupted = model.train(data, goal, args['epochs'], None)
        if interrupted:
            break

        # save results
        key = 'attempt_{}'.format(i)
        arrays_to_save[key] = model.evaluate(data, goal, display=False)
        loss = arrays_to_save[key][-1]
        print("finished {}, with loss {}".format(i, loss))
        np.savez(outfile, **arrays_to_save)


def main():
    np.set_printoptions(precision=6, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset (txt file)")
    parser.add_argument("sdf", help="sdf and gradient of the environment (npz file)")
    parser.add_argument("num-training-attempts", type=int, help="number of times to run train", default=100)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    parser.add_argument("-M", help="dimensions in latent state o_d", type=int, default=2)
    parser.add_argument("-L", help="dimensions in control input", type=int, default=2)
    parser.add_argument("-P", help="dimensions in latent state o_k", type=int, default=2)
    parser.add_argument("-Q", help="dimensions in constraint checking output space", type=int, default=1)
    parser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=10000)

    args = parser.parse_args()
    commandline = ' '.join(sys.argv)
    args.commandline = commandline

    train(args)


if __name__ == '__main__':
    main()
