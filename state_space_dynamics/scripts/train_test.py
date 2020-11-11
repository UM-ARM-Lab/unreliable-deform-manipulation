#!/usr/bin/env python
import argparse
import pathlib

import colorama
import numpy as np
import tensorflow as tf

import rospy
from state_space_dynamics import train_test


# limit_gpu_mem(10)


def train_main(args, seed: int):
    train_test.train_main(dataset_dirs=args.dataset_dirs,
                          model_hparams=args.model_hparams,
                          checkpoint=args.checkpoint,
                          log=args.log,
                          batch_size=args.batch_size,
                          epochs=args.epochs,
                          seed=seed,
                          ensemble_idx=args.ensemble_idx,
                          trials_directory=pathlib.Path('trials'),
                          use_gt_rope=args.use_gt_rope,
                          take=args.take,
                          )


def eval_main(args, seed: int):
    train_test.eval_main(args.dataset_dirs, args.checkpoint, args.mode, args.batch_size, args.use_gt_rope)


def viz_main(args, seed: int):
    train_test.viz_main(**vars(args))


def main():
    colorama.init(autoreset=True)

    np.set_printoptions(linewidth=250, precision=4, suppress=True, threshold=10000)
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--take', type=int)
    train_parser.add_argument('--epochs', type=int, default=500)
    train_parser.add_argument('--ensemble-idx', type=int)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--log-scalars-every', type=int, help='loss/accuracy every this many steps/batches',
                              default=100)
    train_parser.add_argument('--validation-every', type=int, help='report validation every this many epochs',
                              default=1)
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.add_argument('--use-gt-rope', action='store_true')
    train_parser.set_defaults(func=train_main)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    eval_parser.add_argument('--batch-size', type=int, default=32)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.add_argument('--seed', type=int, default=None)
    eval_parser.add_argument('--use-gt-rope', action='store_true')
    eval_parser.set_defaults(func=eval_main)

    viz_parser = subparsers.add_parser('viz')
    viz_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    viz_parser.add_argument('checkpoint', type=pathlib.Path)
    viz_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    viz_parser.add_argument('--verbose', '-v', action='count', default=0)
    viz_parser.add_argument('--seed', type=int, default=None)
    viz_parser.add_argument('--use-gt-rope', action='store_true')
    viz_parser.set_defaults(func=viz_main)

    args = parser.parse_args()

    from time import time
    now = str(int(time()))
    name = f"train_test_{now}"
    rospy.init_node(name)

    if args.seed is None:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    print("Using seed {}".format(seed))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args, seed)


if __name__ == '__main__':
    main()
