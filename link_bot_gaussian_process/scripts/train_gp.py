#!/usr/bin/env python

import argparse

import gpflow as gpf
import numpy as np
import tensorflow as tf

from link_bot_gaussian_process import link_bot_gp, data_reformatting


def main():
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('--verbose')

    args = parser.parse_args()

    data = np.load(args.data)
    train_idx_start = 0
    train_idx_end = 200
    take_every = 1

    train_data = data_reformatting.format_forward_data(data, train_idx_start, train_idx_end, take_every)
    train_x_flat, train_y_flat, train_u_flat, combined_train_x, train_x_trajs, train_u_trajs = train_data

    gpf.reset_default_graph_and_session()

    fwd_model = link_bot_gp.LinkBotGP(combined_train_x, train_y_flat)
    fwd_model.train(verbose=args.verbose)
    fwd_model.save()


if __name__ == '__main__':
    main()
