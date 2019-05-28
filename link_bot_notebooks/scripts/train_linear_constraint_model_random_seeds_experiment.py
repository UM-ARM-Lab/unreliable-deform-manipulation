#!/usr/bin/env python
from __future__ import print_function

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Fore

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
    args['random_init'] = True
    N = 6
    M = 2
    L = 2
    P = 2
    Q = 1

    goal = np.zeros((1, N))
    arrays_to_save = {}
    name = experiments_util.experiment_name("attempt_train")
    outfile = os.path.join('test_data', name + ".npz")
    print(Fore.CYAN, "Saving results to: ", outfile, Fore.RESET)

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

        # if the model is good enough, checkpoint it
        if loss < args['ckpt_loss_threshold']:
            experiment_name = experiments_util.experiment_name('random_init_linear_constraint_model',
                                                               'seed_{}'.format(i))
            log_path = os.path.join(os.getcwd(), "log_data", experiment_name)
            experiments_util.make_log_dir(log_path)
            full_log_path = os.path.join(os.getcwd(), "log_data", log_path)
            model.save(full_log_path, loss=True)


def plot(args):
    data = np.load(args.data)
    evaluation_results = data.values()

    print("# Evaluations: {}".format(len(evaluation_results)))

    total_losses = [e[-1] for e in evaluation_results]
    mean_total_loss = np.mean(total_losses)
    std_total_loss = np.std(total_losses)
    min_total_loss = np.min(total_losses)
    max_total_loss = np.max(total_losses)
    sorted_idx = np.argsort(total_losses)
    sorted_losses = [total_losses[i] for i in sorted_idx]
    all_sorted = [evaluation_results[i] for i in sorted_idx]
    best_params = all_sorted[0]

    print("Mean total loss: {:0.3f}".format(mean_total_loss))
    print("Stdev total loss: {:0.3f}".format(std_total_loss))
    print("Min total loss: {:0.3f}".format(min_total_loss))
    print("Max total loss: {:0.3f}".format(max_total_loss))

    print("Best parameters:")
    # R_d, A_d, B_d, D, R_k, A_k, B_k, threshold_k, spd_loss, spk_loss, c_loss, k_loss, reg, loss
    print("R_d:\n{}".format(best_params[0]))
    print("A_d:\n{}".format(best_params[1]))
    print("B_d:\n{}".format(best_params[2]))
    print("R_k:\n{}".format(best_params[3]))
    print("A_k:\n{}".format(best_params[4]))
    print("B_k:\n{}".format(best_params[5]))
    print("threshold_k:\n{}".format(best_params[6]))
    print("State Prediction Loss in d: {}".format(best_params[7]))
    print("State Prediction Loss in k: {}".format(best_params[8]))
    print("Cost Loss: {}".format(best_params[9]))
    print("Constraint Loss: {}".format(best_params[10]))
    print("Regularization: {}".format(best_params[11]))
    print("Overall Loss: {}".format(best_params[12]))

    plt.hist(total_losses, bins=np.arange(0, 8, 0.2))
    plt.xlabel("Loss after 10,000 training steps")
    plt.ylabel("count")
    plt.show()


def main():
    np.set_printoptions(precision=6, suppress=True)

    plt.style.use("paper")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    gen_subparser = subparsers.add_parser('generate')
    gen_subparser.add_argument("dataset", help="dataset (txt file)")
    gen_subparser.add_argument("sdf", help="sdf and gradient of the environment (npz file)")
    gen_subparser.add_argument("num-training-attempts", type=int, help="number of times to run train")
    gen_subparser.add_argument("--verbose", action='store_true')
    gen_subparser.add_argument("-N", help="dimensions in input state", type=int, default=6)
    gen_subparser.add_argument("-M", help="dimensions in latent state o_d", type=int, default=2)
    gen_subparser.add_argument("-L", help="dimensions in control input", type=int, default=2)
    gen_subparser.add_argument("-P", help="dimensions in latent state o_k", type=int, default=2)
    gen_subparser.add_argument("-Q", help="dimensions in constraint checking output space", type=int, default=1)
    gen_subparser.add_argument("--epochs", "-e", type=int, help="number of epochs to train for", default=10000)
    gen_subparser.add_argument("--seed-offset", "-s", type=int, help="offset the random seed", default=100)
    gen_subparser.add_argument("--ckpt-loss-threshold", type=float, default=0.9)
    gen_subparser.set_defaults(func=train)

    plot_subparser = subparsers.add_parser("plot")
    plot_subparser.add_argument('data', help='npz file of generated results')
    plot_subparser.set_defaults(func=plot)

    args = parser.parse_args()
    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
