#!/usr/bin/env python

import argparse
from tabulate import tabulate

import gpflow as gpf
from colorama import Fore
import numpy as np
import tensorflow as tf

from link_bot_gaussian_process import link_bot_gp, data_reformatting
from link_bot_notebooks import experiments_util
from link_bot_notebooks import toy_problem_optimization_common as tpoc


def make_row(metric_name, e):
    return [metric_name, np.min(e), np.max(e), np.mean(e), np.median(e), np.std(e)]


def fwd_model_error_metrics(my_model, test_x, test_y):
    """
    compute the euclidian distance for each node in pred_y[i] to each node in test_y[i],
    averaged over all i using the max likelihood prediction
    """
    pred_delta_x_mean, pred_delta_x_sigma = my_model.model.predict_y(test_x)
    tail_error = np.linalg.norm(pred_delta_x_mean[:, 0:2] - test_y[:, 0:2], axis=1)
    mid_error = np.linalg.norm(pred_delta_x_mean[:, 2:4] - test_y[:, 2:4], axis=1)
    head_error = np.linalg.norm(pred_delta_x_mean[:, 4:6] - test_y[:, 4:6], axis=1)
    total_node_error = tail_error + mid_error + head_error
    # each column goes [metric name, min, max, mean, median, std]
    return np.array([make_row('tail position error (m)', tail_error),
                     make_row('mid position error (m)', mid_error),
                     make_row('head position error (m)', head_error),
                     make_row('total position error (m)', total_node_error)], dtype=np.object)


def inv_model_error_metrics(my_model, test_x, test_y):
    """ compute the euclidean distance between the predicted control and the true control"""
    pred_u_mean, pred_u_sigma = my_model.model.predict_y(test_x)
    pred_speeds = abs(pred_u_mean[:, 2])

    abs_speed_error = abs(pred_speeds - abs(test_y[:, 2]))

    # compute dot product of each column of a with each column of b
    pred_theta = np.arctan2(pred_u_mean[:, 1], pred_u_mean[:, 0])
    true_theta = np.arctan2(test_y[:, 1], test_y[:, 0])
    abs_angle_error = abs(np.rad2deg(tpoc.yaw_diff(true_theta, pred_theta)))

    return np.array([make_row('speed (m/s)', abs_speed_error),
                     make_row('angle (deg)', abs_angle_error)], dtype=np.object)


def main():
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dont-save', action='store_true')

    args = parser.parse_args()

    # Load data
    ###########################################################################
    data = np.load(args.data)
    train_idx_start = 0
    train_idx_end = 200
    test_idx_start = 200
    test_idx_end = 240

    fwd_train_data = data_reformatting.format_forward_data(data, train_idx_start, train_idx_end)
    fwd_train_x = fwd_train_data[3]
    fwd_train_y = fwd_train_data[1]
    fwd_test_data = data_reformatting.format_forward_data(data, test_idx_start, test_idx_end)
    fwd_test_x = fwd_test_data[3]
    fwd_test_y = fwd_test_data[1]

    inv_train_data = data_reformatting.format_inverse_data(data, train_idx_start, train_idx_end, take_every=10)
    inv_train_x = inv_train_data[0]
    inv_train_y = inv_train_data[1]
    inv_test_data = data_reformatting.format_inverse_data(data, test_idx_start, test_idx_end, take_every=10)
    inv_test_x = inv_test_data[0]
    inv_test_y = inv_test_data[1]

    # Train
    ###########################################################################
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.1))
    gpf.reset_default_session(config=config)
    fwd_model = link_bot_gp.LinkBotGP()
    inv_model = link_bot_gp.LinkBotGP()

    print("Training forward model")
    fwd_model.train(fwd_train_x, fwd_train_y, verbose=args.verbose)
    print("Training inverse model")
    inv_model.train(inv_train_x, inv_train_y, verbose=args.verbose)

    # Save
    ###########################################################################
    if not args.dont_save:
        log_path = experiments_util.experiment_name('separate_independent', 'gpf')
        fwd_model.save(log_path, 'fwd_model')
        inv_model.save(log_path, 'inv_model')

    # Evaluate
    ###########################################################################
    headers = ['error metric', 'min', 'max', 'mean', 'median', 'std']
    aggregate_metrics = np.vstack((fwd_model_error_metrics(fwd_model, fwd_test_x, fwd_test_y),
                                   inv_model_error_metrics(inv_model, inv_test_x, inv_test_y)))
    table = tabulate(aggregate_metrics, headers=headers, tablefmt='github', floatfmt='6.3f')
    print(table)
    with open("metrics.md", 'w') as f:
        f.writelines(table)


if __name__ == '__main__':
    main()
