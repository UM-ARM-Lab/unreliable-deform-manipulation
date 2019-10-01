#!/usr/bin/env python

import argparse
import json
import os
import pathlib

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Fore, Style
from tabulate import tabulate

from link_bot_gaussian_process import link_bot_gp, error_metrics, data_reformatting
from link_bot_pycommon import experiments_util
from video_prediction.datasets import dataset_utils


def load_data(sess, indir, dataset_hparams_dict, mode, n_examples, sequence_length=2, seed=0):
    dataset, inputs, _ = dataset_utils.get_inputs(indir,
                                                  'state_space',
                                                  dataset_hparams_dict,
                                                  'sequence_length={}'.format(sequence_length),
                                                  mode=mode,
                                                  epochs=1,
                                                  seed=seed,
                                                  batch_size=n_examples)

    try:
        data_x, data_y = sess.run(inputs)
    except tf.errors.OutOfRangeError as e:
        print(Fore.RED + "{} Dataset does not contain {} examples.".format(mode, n_examples) + Fore.RESET)
        raise e

    fwd_data = data_reformatting.format_forward_data_gz_tfrecords(data_x['states'], data_y['output_states'], data_x['actions'])
    gp_x, gp_y = fwd_data

    return data_x, data_y, gp_x, gp_y


def train(args):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.1))
    gpf.reset_default_session(config=config)
    sess = gpf.get_default_session()

    fwd_model = link_bot_gp.LinkBotGP()

    num_validation_examples = 100

    if args.dataset_hparams_dict:
        dataset_hparams_dict = json.load(args.dataset_hparams_dict.open('r'))
    else:
        dataset_hparams_dict = json.load((args.indir / 'hparams.json').open('r'))

    train_x, train_y, train_fwd_gp_x, train_fwd_gp_y = load_data(sess,
                                                                 args.indir,
                                                                 dataset_hparams_dict,
                                                                 'train',
                                                                 args.n_training_examples,
                                                                 seed=args.seed)
    val_x, val_y, val_fwd_gp_x, val_fwd_gp_y = load_data(sess,
                                                         args.indir,
                                                         dataset_hparams_dict,
                                                         'val',
                                                         num_validation_examples,
                                                         seed=args.seed)

    # Train
    ###########################################################################

    print(Fore.CYAN + "Training forward model" + Fore.RESET)
    fwd_model.train(train_fwd_gp_x,
                    train_fwd_gp_y,
                    beta=args.beta,
                    verbose=args.verbose,
                    maximum_training_iterations=args.max_iters,
                    n_inducing_points=args.n_inducing_points,
                    dataset_hparams=dataset_hparams_dict)

    # Save
    ###########################################################################
    if not args.dont_save:
        log_path = experiments_util.experiment_name(args.log, 'gpf')
        fwd_model.save(log_path, 'fwd_model')

    print(Fore.CYAN + "Evaluating" + Fore.RESET)

    evaluate(fwd_model, val_fwd_gp_x, val_fwd_gp_y)


def eval(args):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.1))
    gpf.reset_default_session(config=config)
    sess = gpf.get_default_session()

    fwd_model_path = args.model_dir / "fwd_model"
    fwd_model = link_bot_gp.LinkBotGP()
    fwd_model.load(fwd_model_path)

    if args.dataset_hparams_dict:
        dataset_hparams_dict = json.load(args.dataset_hparams_dict.open('r'))
    else:
        dataset_hparams_dict = json.load((args.indir / 'hparams.json').open('r'))

    num_test_examples = 128

    test_x, test_y, test_fwd_gp_x, test_fwd_gp_y = load_data(sess, args.indir, dataset_hparams_dict, 'test', num_test_examples,
                                                             seed=args.seed)
    long_test_x, long_test_y, long_test_fwd_gp_x, long_test_fwd_gp_y = load_data(sess, args.indir, dataset_hparams_dict, 'test',
                                                                                 num_test_examples, sequence_length=10,
                                                                                 seed=args.seed)
    evaluate(fwd_model, test_fwd_gp_x, test_fwd_gp_y)
    multistep_evaluate(fwd_model, long_test_x, long_test_y)

    if not args.no_plot:
        visualize(fwd_model, long_test_x, long_test_y)


def visualize(fwd_model, data_x, data_y):
    x = data_x['states']
    y = data_y['output_states']
    a = data_x['actions']
    for i, (xi, yi, ai) in enumerate(zip(x, y, a)):
        x0 = np.expand_dims(xi[0], 0)
        prediction, _ = link_bot_gp.predict(fwd_model, x0, ai)
        _ = link_bot_gp.animate_predict(prediction, yi, extent=[-2, 2, -2, 2], sdf=None, linewidth=2, example_idx=i)
        plt.show()


def multistep_evaluate(fwd_model, fwd_test_x, fwd_test_y):
    headers = ['error metric', 'min', 'max', 'mean', 'median', 'std']
    aggregate_metrics = error_metrics.multistep_fwd_model_error_metrics(fwd_model, fwd_test_x, fwd_test_y)
    table = tabulate(aggregate_metrics, headers=headers, tablefmt='github', floatfmt='6.4f')
    print(Style.BRIGHT + "Multi-Step Error:" + Style.RESET_ALL)
    print(table)


def evaluate(fwd_model, fwd_test_x, fwd_test_y):
    headers = ['error metric', 'min', 'max', 'mean', 'median', 'std']
    aggregate_metrics = error_metrics.fwd_model_error_metrics(fwd_model, fwd_test_x, fwd_test_y)
    table = tabulate(aggregate_metrics, headers=headers, tablefmt='github', floatfmt='6.3f')
    print(table)
    with open("metrics.md", 'w') as f:
        f.writelines(table)
        f.write("\n")


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=1000)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('indir', type=pathlib.Path)
    train_parser.add_argument('--dataset-hparams-dict')
    train_parser.add_argument('--seed', type=int, default=0)
    train_parser.add_argument('--n-training-examples', type=int, default=1000)
    train_parser.add_argument('--max-iters', type=int, default=1000)
    train_parser.add_argument('--beta', type=float, default=1000)
    train_parser.add_argument('--n-inducing-points', type=int, default=20)
    train_parser.add_argument('--verbose', action='store_true')
    train_parser.add_argument('--log')
    train_parser.add_argument('--dont-save', action='store_true')
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('indir', type=pathlib.Path)
    eval_parser.add_argument('model_dir', type=pathlib.Path)
    eval_parser.add_argument('--dataset-hparams-dict')
    eval_parser.add_argument('--seed', type=int, default=0)
    eval_parser.add_argument('--no-plot', action='store_true')
    eval_parser.add_argument('--verbose', action='store_true')
    eval_parser.set_defaults(func=eval)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    args.func(args)


if __name__ == '__main__':
    main()
