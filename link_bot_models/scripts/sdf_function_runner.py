#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import tensorflow as tf

from link_bot_models import base_model
from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset
from link_bot_models.sdf_function_model import SDFFunctionModelRunner
from link_bot_pycommon import experiments_util


def train(args):
    log_path = experiments_util.experiment_name(args.log)

    train_dataset = MultiEnvironmentDataset.load_dataset(args.train_dataset)
    validation_dataset = MultiEnvironmentDataset.load_dataset(args.validation_dataset)
    sdf_shape = train_dataset.sdf_shape

    if args.checkpoint:
        model = SDFFunctionModelRunner.load(args.checkpoint)
    else:
        args_dict = {
            'sdf_shape': sdf_shape,
            'beta': 1e-2,
            'fc_layer_sizes': [32, 32],
            'sigmoid_scale': 100,
            'N': train_dataset.N
        }
        args_dict.update(base_model.make_args_dict(args))
        model = SDFFunctionModelRunner(args_dict)

    model.train(train_dataset, validation_dataset, args.label_types_map, log_path, args)


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=220)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser, train_subparser, eval_subparser, show_subparser = base_model.base_parser()

    train_subparser.set_defaults(func=train)
    eval_subparser.set_defaults(func=SDFFunctionModelRunner.evaluate_main)
    show_subparser.set_defaults(func=SDFFunctionModelRunner.show)

    parser.run()


if __name__ == '__main__':
    main()
