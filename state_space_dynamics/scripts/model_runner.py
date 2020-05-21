#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf

import state_space_dynamics
from link_bot_data.dynamics_dataset import DynamicsDataset
from link_bot_pycommon.get_scenario import get_scenario
from moonshine import experiments_util
from moonshine.gpu_config import limit_gpu_mem
from moonshine.tensorflow_train_test_loop import evaluate, train

limit_gpu_mem(6)


def train_func(args, seed: int):
    if args.log:
        if args.ensemble_idx is not None:
            log_path = pathlib.Path(args.log) / str(args.ensemble_idx)
        else:
            log_path = experiments_util.experiment_name(args.log)
    else:
        log_path = None

    # Model parameters
    model_hparams = json.load(args.model_hparams.open('r'))

    # Datasets
    train_dataset = DynamicsDataset(args.dataset_dirs)
    train_tf_dataset = train_dataset.get_datasets(mode='train', sequence_length=model_hparams['sequence_length'])
    val_dataset = DynamicsDataset(args.dataset_dirs)
    val_tf_dataset = val_dataset.get_datasets(mode='val', sequence_length=model_hparams['sequence_length'])
    train_tf_dataset = train_tf_dataset.shuffle(seed=args.seed, buffer_size=1024).batch(args.batch_size, drop_remainder=True)
    val_tf_dataset = val_tf_dataset.batch(args.batch_size, drop_remainder=True)

    train_tf_dataset = train_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_tf_dataset = val_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Copy parameters of the dataset into the model
    model_hparams['dynamics_dataset_hparams'] = train_dataset.hparams
    model_hparams['batch_size'] = args.batch_size
    model = state_space_dynamics.get_model(model_hparams['model_class'])
    scenario = get_scenario(train_dataset.hparams['scenario'])
    net = model(hparams=model_hparams, batch_size=args.batch_size, scenario=scenario)

    ###############
    # Train
    ###############
    train(keras_model=net,
          model_hparams=model_hparams,
          train_tf_dataset=train_tf_dataset,
          val_tf_dataset=val_tf_dataset,
          dataset_dirs=args.dataset_dirs,
          seed=seed,
          batch_size=args.batch_size,
          epochs=model_hparams['epochs'],
          loss_function=scenario.dynamics_loss_function,
          metrics_function=scenario.dynamics_metrics_function,
          checkpoint=args.checkpoint,
          log_path=log_path,
          log_scalars_every=args.log_scalars_every)


def eval_func(args, seed: int):
    ###############
    # Dataset
    ###############
    test_dataset = DynamicsDataset(args.dataset_dirs)
    test_tf_dataset = test_dataset.get_datasets(mode=args.mode,
                                                sequence_length=args.sequence_length,
                                                )

    test_tf_dataset = test_tf_dataset.batch(args.batch_size, drop_remainder=True)

    ###############
    # Model
    ###############
    model_hparams_file = args.checkpoint / 'hparams.json'
    model_hparams = json.load(open(model_hparams_file, 'r'))
    model_hparams['dt'] = test_dataset.hparams['dt']

    model = state_space_dynamics.get_model(model_hparams['model_class'])
    scenario = get_scenario(test_dataset.hparams['scenario'])
    net = model(hparams=model_hparams, batch_size=args.batch_size, scenario=scenario)

    ###############
    # Evaluate
    ###############
    evaluate(keras_model=net,
             test_tf_dataset=test_tf_dataset,
             loss_function=scenario.dynamics_loss_function,
             metrics_function=scenario.dynamics_metrics_function,
             checkpoint_path=args.checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--ensemble-idx', type=int)
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--log-scalars-every', type=int, help='loss/accuracy every this many steps/batches', default=119)
    train_parser.set_defaults(func=train_func)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--sequence-length', type=int, default=10)
    eval_parser.add_argument('--batch-size', type=int, default=32)
    eval_parser.add_argument('--mode', type=str, choices=['test', 'val', 'train'], default='test')
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.set_defaults(func=eval_func)

    args = parser.parse_args()

    if args.seed is None:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    print("Random seed: {}".format(seed))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args, seed)


if __name__ == '__main__':
    main()
