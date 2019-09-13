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
    train_parser.add_argument('dataset_hparams_dict', type=pathlib.Path)
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--dataset-hparams', type=str)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--log', '-l')

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

    train_dataset, train_tf_dataset = dataset_utils.get_dataset(args.input_dir,
                                                                'state_space',
                                                                args.dataset_hparams_dict,
                                                                args.dataset_hparams,
                                                                mode='train',
                                                                epochs=args.epochs,
                                                                seed=args.seed,
                                                                batch_size=args.batch_size)
    d = train_tf_dataset.make_one_shot_iterator().get_next()
    print(d.keys())
    print(d['actions'].shape)
    return

    val_dataset, val_tf_dataset = dataset_utils.get_dataset(args.input_dir,
                                                            'state_space',
                                                            args.dataset_hparams_dict,
                                                            args.dataset_hparams,
                                                            mode='val',
                                                            epochs=None,
                                                            seed=args.seed,
                                                            batch_size=args.batch_size)

    # Now that we have the input tensors, so we can construct our Keras model
    if args.checkpoint:
        model = LocallyLinearModel.load(args.checkpoint)
    else:
        args_dict = {
            'sdf_shape': train_dataset.hparams.sdf_shape,
            'conv_filters': [
                (8, (5, 5)),
                (8, (5, 5)),
            ],
            'fc_layer_sizes': [32, 32],
            'N': train_dataset.hparams.rope_config_dim,
        }
        model_hparams = json.load(open(args.model_hparams, 'r'))
        model = LocallyLinearModel(model_hparams)

    try:
        model.train(train_dataset, train_tf_dataset, val_dataset, val_tf_dataset, log_path, args)
    except KeyboardInterrupt:
        print("Interrupted.")
        pass


if __name__ == '__main__':
    main()
