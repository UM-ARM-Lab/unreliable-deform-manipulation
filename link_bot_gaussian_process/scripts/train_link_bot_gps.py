#!/usr/bin/env python

import argparse
import os

import gpflow as gpf
import numpy as np
import tensorflow as tf
from tabulate import tabulate

from link_bot_data.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_gaussian_process import link_bot_gp, data_reformatting, error_metrics
from link_bot_models.label_types import LabelType
from link_bot_pycommon import experiments_util


def train(args):
    # Load data
    ###########################################################################
    train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    test_dataset = MultiEnvironmentDataset.load_dataset(args.test_dataset)

    fwd_train_data = data_reformatting.format_forward_data_gz(args, train_dataset)
    fwd_train_x = fwd_train_data[3]
    fwd_train_y = fwd_train_data[1]
    fwd_test_data = data_reformatting.format_forward_data_gz(args, test_dataset)
    fwd_test_x = fwd_test_data[3]
    fwd_test_y = fwd_test_data[1]

    inv_train_data = data_reformatting.format_inverse_data_gz(args, train_dataset)
    inv_train_x = inv_train_data[0]
    inv_train_y = inv_train_data[1]
    inv_test_data = data_reformatting.format_inverse_data_gz(args, test_dataset)
    inv_test_x = inv_test_data[0]
    inv_test_y = inv_test_data[1]

    # anim = link_bot_gp.animate_training_data(fwd_train_data[-2])
    # plt.show()

    # Train
    ###########################################################################
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.1))
    gpf.reset_default_session(config=config)
    fwd_model = link_bot_gp.LinkBotGP()
    inv_model = link_bot_gp.LinkBotGP()

    print("Training forward model")
    fwd_model.train(fwd_train_x, fwd_train_y, verbose=args.verbose, maximum_training_iterations=500,
                    n_inducing_points=500)
    print("Training inverse model")
    inv_model.train(inv_train_x, inv_train_y, verbose=args.verbose, maximum_training_iterations=500,
                    n_inducing_points=50)

    # Save
    ###########################################################################
    if not args.dont_save:
        log_path = experiments_util.experiment_name('separate_independent', 'gpf')
        fwd_model.save(log_path, 'fwd_model')
        inv_model.save(log_path, 'inv_model')

    evaluate(args, fwd_model, inv_model, fwd_test_x, fwd_test_y, inv_test_x, inv_test_y)


def eval(args):
    fwd_model = link_bot_gp.LinkBotGP()
    inv_model = link_bot_gp.LinkBotGP()

    fwd_model_path = os.path.join(args.model_dir, "fwd_model")
    inv_model_path = os.path.join(args.model_dir, "inv_model")

    fwd_model.load(fwd_model_path)
    inv_model.load(inv_model_path)

    test_dataset = MultiEnvironmentDataset.load_dataset(args.test_dataset)

    fwd_test_data = data_reformatting.format_forward_data_gz(args, test_dataset)
    fwd_test_x = fwd_test_data[3]
    fwd_test_y = fwd_test_data[1]
    inv_test_data = data_reformatting.format_inverse_data_gz(args, test_dataset)
    inv_test_x = inv_test_data[0]
    inv_test_y = inv_test_data[1]

    evaluate(args, fwd_model, inv_model, fwd_test_x, fwd_test_y, inv_test_x, inv_test_y)


def evaluate(args, fwd_model, inv_model, fwd_test_x, fwd_test_y, inv_test_x, inv_test_y):
    headers = ['error metric', 'min', 'max', 'mean', 'median', 'std']
    aggregate_metrics = np.vstack((error_metrics.fwd_model_error_metrics(fwd_model, fwd_test_x, fwd_test_y),
                                   error_metrics.inv_model_error_metrics(inv_model, inv_test_x, inv_test_y)))
    table = tabulate(aggregate_metrics, headers=headers, tablefmt='github', floatfmt='6.3f')
    print(table)
    with open("metrics.md", 'w') as f:
        f.writelines(table)


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=1000)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('train_dataset')
    train_parser.add_argument('test_dataset')
    train_parser.add_argument("mask_label_type", type=LabelType.from_string)
    train_parser.add_argument('--seed', type=int, default=0)
    train_parser.add_argument('--verbose', action='store_true')
    train_parser.add_argument('--dont-save', action='store_true')
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('test_dataset')
    eval_parser.add_argument('model_dir')
    eval_parser.add_argument("mask_label_type", type=LabelType.from_string)
    eval_parser.add_argument('--seed', type=int, default=0)
    eval_parser.add_argument('--verbose', action='store_true')
    eval_parser.set_defaults(func=eval)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    args.func(args)


if __name__ == '__main__':
    main()
