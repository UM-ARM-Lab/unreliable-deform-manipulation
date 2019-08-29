#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import tensorflow as tf

from link_bot_models import base_model_runner
from link_bot_models.raster_cnn_model import RasterCNNModelRunner
from link_bot_pycommon import experiments_util
from video_prediction.datasets import dataset_utils


def train(args):
    if args.log:
        log_path = experiments_util.experiment_name(args.log)
    else:
        log_path = None

    train_dataset, train_inputs = dataset_utils.get_inputs(args.input_dir,
                                                           args.dataset,
                                                           args.dataset_hparams_dict,
                                                           args.dataset_hparams,
                                                           mode='train',
                                                           epochs=args.epochs,
                                                           seed=args.seed,
                                                           batch_size=args.batch_size)
    val_dataset, val_inputs = dataset_utils.get_inputs(args.input_dir,
                                                       args.dataset,
                                                       args.dataset_hparams_dict,
                                                       args.dataset_hparams,
                                                       mode='val',
                                                       epochs=1,
                                                       seed=args.seed,
                                                       batch_size=args.batch_size)

    # Now that we have the input tensors, so we can construct our Keras model
    if args.checkpoint:
        train_model = RasterCNNModelRunner.load(args.checkpoint, train_inputs)
    else:
        args_dict = {
            'sdf_shape': train_dataset.hparams.sdf_shape,
            'conv_filters': [
                (32, (5, 5)),
                (32, (5, 5)),
                (16, (3, 3)),
                (16, (3, 3)),
            ],
            'fc_layer_sizes': [256, 256],
            'N': train_dataset.hparams.rope_config_dim,
        }
        args_dict.update(base_model_runner.make_args_dict(args))
        train_model = RasterCNNModelRunner(args_dict, train_inputs)
        val_model = RasterCNNModelRunner(args_dict, val_inputs)

    train_model.train(train_dataset, log_path, args)


def main():
    np.set_printoptions(precision=6, suppress=True)
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser, train_subparser, eval_subparser, show_subparser = base_model_runner.base_parser()

    train_subparser.set_defaults(func=train)
    eval_subparser.set_defaults(func=RasterCNNModelRunner.evaluate_main)
    show_subparser.set_defaults(func=RasterCNNModelRunner.show)

    parser.run()


if __name__ == '__main__':
    main()
