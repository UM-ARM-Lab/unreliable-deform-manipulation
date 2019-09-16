#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf

from link_bot_pycommon import experiments_util
from state_space_dynamics.locally_linear_model import LocallyLinearModel
from video_prediction.datasets import dataset_utils

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('input_dir', type=pathlib.Path)
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--dataset-hparams-dict', type=pathlib.Path)
    train_parser.add_argument('--dataset-hparams', type=str)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--validation', action='store_true')
    train_parser.add_argument('--debug', action='store_true')

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('checkpoint')

    args = parser.parse_args()

    if args.seed is None:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    np.random.seed(seed)
    tf.random.set_random_seed(seed)

    if args.log:
        log_path = experiments_util.experiment_name(args.log)
    else:
        log_path = None

    if args.dataset_hparams_dict:
        dataset_hparams_dict = json.load(open(args.dataset_hparams_dict, 'r'))
    else:
        dataset_hparams_dict = json.load(open(args.input_dir / 'hparams.json', 'r'))

    model_hparams = json.load(open(args.model_hparams, 'r'))
    dataset_hparams_dict['sequence_length'] = model_hparams['sequence_length']
    dataset_hparams_dict['sdf_shape'] = model_hparams['sdf_shape']

    ###############
    # Datasets
    ###############
    train_dataset, train_tf_dataset = dataset_utils.get_dataset(args.input_dir,
                                                                'state_space',
                                                                dataset_hparams_dict,
                                                                args.dataset_hparams,
                                                                mode='train',
                                                                epochs=args.epochs,
                                                                seed=args.seed,
                                                                batch_size=args.batch_size)
    val_dataset, val_tf_dataset = dataset_utils.get_dataset(args.input_dir,
                                                            'state_space',
                                                            dataset_hparams_dict,
                                                            args.dataset_hparams,
                                                            mode='val',
                                                            epochs=None,
                                                            seed=args.seed,
                                                            batch_size=args.batch_size)

    ###############
    # Model
    ###############
    if args.checkpoint:
        model = LocallyLinearModel.load(args.checkpoint)
    else:
        model = LocallyLinearModel(model_hparams)

    try:
        ###############
        # Train
        ###############
        model.train(train_dataset, train_tf_dataset, val_dataset, val_tf_dataset, log_path, args)
    except KeyboardInterrupt:
        print("Interrupted.")
        pass


if __name__ == '__main__':
    main()
