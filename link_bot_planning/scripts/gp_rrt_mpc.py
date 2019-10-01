#!/usr/bin/env python
from __future__ import division, print_function

import argparse
import os
import pathlib
import time

import numpy as np
import ompl.util as ou
import tensorflow as tf
from colorama import Fore

from link_bot_gaussian_process import link_bot_gp
from link_bot_planning import shooting_rrt_mpc

tf.enable_eager_execution()


def test(args):
    args.n_trials = 1
    fwd_gp_model = link_bot_gp.LinkBotGP(ou.RNG)
    fwd_gp_model.load(args.gp_model_dir / 'fwd_model')
    dt = fwd_gp_model.dataset_hparams['dt']
    shooting_rrt_mpc.plan_and_execute_random_goals(args, fwd_gp_model, dt)


def eval(args):
    stats_filename = os.path.join(args.gp_model_dir, 'eval_{}.txt'.format(int(time.time())))

    fwd_gp_model = link_bot_gp.LinkBotGP(ou.RNG)
    fwd_gp_model.load(args.gp_model_dir / 'fwd_model')
    dt = fwd_gp_model.dataset_hparams['dt']

    results = shooting_rrt_mpc.plan_and_execute_random_goals(args, fwd_gp_model, dt)
    min_costs, execution_times, planning_times, n_fails, n_successes = results

    eval_stats_lines = [
        '% fail: {}'.format(float(n_fails) / args.n_trials),
        '% success: {}'.format(float(n_successes) / args.n_trials),
        'mean min dist to goal: {}'.format(np.mean(min_costs)),
        'std min dist to goal: {}'.format(np.std(min_costs)),
        'mean planning time: {}'.format(np.mean(planning_times)),
        'std planning time: {}'.format(np.std(planning_times)),
        'mean execution time: {}'.format(np.mean(execution_times)),
        'std execution time: {}'.format(np.std(execution_times)),
        'full data',
        'min costs: {}'.format(np.array2string(min_costs)),
        'execution times: {}'.format(np.array2string(execution_times)),
        '\n'
    ]

    print(eval_stats_lines)
    stats_file = open(stats_filename, 'w')
    print(Fore.CYAN + "writing evaluation statistics to: {}".format(stats_filename) + Fore.RESET)
    stats_file.writelines("\n".join(eval_stats_lines))


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=250)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("gp_model_dir", help="load this saved forward model file", type=pathlib.Path)
    parser.add_argument("--n-trials", '-n', type=int, default=20)
    parser.add_argument("--n-actions", '-T', help="number of actions to execute from the plan", type=int, default=-1)
    parser.add_argument("--planner-timeout", help="time in seconds", type=float, default=60.0)
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument("--seed", '-s', type=int, default=1)
    parser.add_argument("--success-threshold", type=int, default=0.1)
    parser.add_argument('--res', '-r', type=float, default=0.01, help='size of cells in meters')
    # Even though the arena is 5m, we need extra padding so that we can request a 1x1 meter local sdf at the corners
    parser.add_argument('--env-w', type=float, default=6)
    parser.add_argument('--env-h', type=float, default=6)
    parser.add_argument('--sdf-w', type=float, default=1.0)
    parser.add_argument('--sdf-h', type=float, default=1.0)
    parser.add_argument('--max-v', type=float, default=0.25)
    parser.add_argument("--real-time-rate", type=float, default=1.0)
    parser.add_argument("--model-name", '-m', default="link_bot")

    subparsers = parser.add_subparsers()
    test_subparser = subparsers.add_parser("test")
    test_subparser.add_argument('--max-steps', type=int, default=10000)
    test_subparser.set_defaults(func=test)

    eval_subparser = subparsers.add_parser("eval")
    eval_subparser.set_defaults(func=eval)

    args = parser.parse_args()

    np.random.seed(args.seed)
    ou.RNG.setSeed(args.seed)
    # ou.setLogLevel(ou.LOG_DEBUG)
    ou.setLogLevel(ou.LOG_ERROR)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
