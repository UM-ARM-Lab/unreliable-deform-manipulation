#!/usr/bin/env python
import argparse
import pathlib

import colorama
import numpy as np
import tensorflow as tf

import rospy
from link_bot_classifiers import train_test_classifier


def train_main(args):
    if args.seed is None:
        args.seed = np.random.randint(0, 10000)

    print("Using seed {}".format(args.seed))
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    trials_directory = pathlib.Path("./trials").absolute()
    train_test_classifier.train_main(trials_directory=trials_directory, **vars(args))


def test_main(args):
    train_test_classifier.test_main(**vars(args))


def eval_main(args):
    train_test_classifier.eval_main(**vars(args))


def eval_ensemble_main(args):
    train_test_classifier.eval_ensemble_main(**vars(args))


def main():
    colorama.init(autoreset=True)

    rospy.init_node("train_test_classifier")

    np.set_printoptions(linewidth=250, precision=4, suppress=True)
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=24)
    train_parser.add_argument('--take', type=int)
    train_parser.add_argument('--debug', action='store_true')
    train_parser.add_argument('--epochs', type=int, default=10)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--ensemble-idx', type=int)
    train_parser.add_argument('--log-scalars-every', type=int,
                              help='loss/accuracy every this many steps/batches', default=100)
    train_parser.add_argument('--validation-every', type=int,
                              help='report validation every this many epochs', default=1)
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.add_argument('--use-gt-rope', action='store_true')
    train_parser.set_defaults(func=train_main)

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    test_parser.add_argument('checkpoint', type=pathlib.Path)
    test_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    test_parser.add_argument('--batch-size', type=int, default=8)
    test_parser.add_argument('--verbose', '-v', action='count', default=0)
    test_parser.add_argument('--take', type=int)
    test_parser.add_argument('--use-gt-rope', action='store_true')
    test_parser.set_defaults(func=test_main)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    eval_parser.add_argument('--batch-size', type=int, default=8)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.add_argument('--only-errors', action='store_true')
    eval_parser.add_argument('--use-gt-rope', action='store_true')
    eval_parser.set_defaults(func=eval_main)

    eval_ensemble_parser = subparsers.add_parser('eval_ensemble')
    eval_ensemble_parser.add_argument('dataset_dir', type=pathlib.Path)
    eval_ensemble_parser.add_argument('checkpoints', type=pathlib.Path, nargs='+')
    eval_ensemble_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    eval_ensemble_parser.add_argument('--batch-size', type=int, default=1)
    eval_ensemble_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_ensemble_parser.add_argument('--only-errors', action='store_true')
    eval_ensemble_parser.add_argument('--use-gt-rope', action='store_true')
    eval_ensemble_parser.set_defaults(func=eval_ensemble_main)

    args = parser.parse_args()

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
