#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import tensorflow as tf

from link_bot_classifiers import base_classifier_runner
from link_bot_classifiers.sdf_function_model import SDFFunctionModelRunner
from link_bot_pycommon import experiments_util


def train(args):
    if args.log:
        log_path = experiments_util.experiment_name(args.log)
    else:
        log_path = None

    # train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    # validation_dataset = MultiEnvironmentDataset.load_dataset(args.validation_dataset)
    # sdf_shape = train_dataset.sdf_shape

    if args.checkpoint:
        model = SDFFunctionModelRunner.load(args.checkpoint)
    else:
        args_dict = {
            'sdf_shape': sdf_shape,
            'beta': 1e-2,
            'fc_layer_sizes': [32, 32],
            'sigmoid_scale': args.sigmoid_scale,
            'N': train_dataset.N
        }
        args_dict.update(base_classifier_runner.make_args_dict(args))
        model = SDFFunctionModelRunner(args_dict)

    model.train(train_dataset, validation_dataset, args.label_types_map, log_path, args)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=220)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser, train_subparser, eval_subparser, show_subparser = base_classifier_runner.base_parser()

    train_subparser.add_argument('--sigmoid-scale', type=float, default=100.0)
    train_subparser.set_defaults(func=train)
    eval_subparser.set_defaults(func=SDFFunctionModelRunner.evaluate_main)
    show_subparser.set_defaults(func=SDFFunctionModelRunner.show)

    parser.run()


if __name__ == '__main__':
    main()
