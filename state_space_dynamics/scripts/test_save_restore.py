#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import progressbar
import tensorflow as tf
from colorama import Fore

import state_space_dynamics
from link_bot_data.link_bot_state_space_dataset import LinkBotStateSpaceDataset

tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def test_save_and_restore(args):
    model_hparams = json.load(open(args.model_hparams, 'r'))

    dataset = LinkBotStateSpaceDataset(args.dataset_dirs)
    tf_dataset = dataset.get_datasets(mode='train',
                                      shuffle=False,
                                      seed=args.seed,
                                      sequence_length=model_hparams['sequence_length'],
                                      batch_size=args.batch_size)

    # Copy parameters of the dataset into the model
    model_hparams['dynamics_dataset_hparams'] = dataset.hparams
    module = state_space_dynamics.get_model_module(model_hparams['model_class'])
    net1 = module.model(hparams=model_hparams)

    optimizer = tf.train.AdamOptimizer()
    loss_fn = tf.keras.losses.MeanSquaredError()
    global_step = tf.train.get_or_create_global_step()

    full_log_path = '/tmp'
    ckpt = tf.train.Checkpoint(step=global_step, optimizer=optimizer, net=net1)
    manager = tf.train.CheckpointManager(ckpt, full_log_path, max_to_keep=1)

    # get a test input
    x, y = next(tf_dataset.take(1).__iter__())
    outputs1 = net1(x)
    weights1 = net1.get_weights()
    loss1 = loss_fn(y_true=y['output_states'], y_pred=outputs1).numpy()

    ################
    # Save
    ################
    _ = manager.save()

    ################
    # Restore
    ################
    net2 = module.model(hparams=model_hparams)
    ckpt = tf.train.Checkpoint(net=net2)
    manager = tf.train.CheckpointManager(ckpt, full_log_path, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    outputs2 = net2(x)
    weights2 = net2.get_weights()
    loss2 = loss_fn(y_true=y['output_states'], y_pred=outputs2).numpy()

    for w1, w2 in zip(weights1, weights2):
        print(np.allclose(w1, w2))
    print(loss1 - loss2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)

    subparsers = parser.add_subparsers()

    parser = subparsers.add_parser('train')
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('model_hparams', type=pathlib.Path)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1)
    parser.set_defaults(func=test_save_and_restore)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    test_save_and_restore(args)


if __name__ == '__main__':
    main()
