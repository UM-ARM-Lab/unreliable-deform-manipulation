#!/usr/bin/env python
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf

import link_bot_classifiers
from link_bot_data.classifier_dataset import ClassifierDataset
from moonshine.image_functions import setup_image_inputs
from link_bot_planning.get_scenario import get_scenario
from moonshine import experiments_util
from moonshine.base_classifier_model import binary_classification_loss_function, binary_classification_metrics_function
from moonshine.tensorflow_train_test_loop import evaluate, train

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train_main(args, seed: int):
    if args.log:
        log_path = experiments_util.experiment_name(args.log)
    else:
        log_path = None

    ###############
    # Datasets
    ###############
    train_dataset = ClassifierDataset(args.dataset_dirs)
    val_dataset = ClassifierDataset(args.dataset_dirs)

    ###############
    # Model
    ###############
    model_hparams = json.load((args.model_hparams).open('r'))
    model_hparams['classifier_dataset_hparams'] = train_dataset.hparams
    model = link_bot_classifiers.get_model(model_hparams['model_class'])
    scenario = get_scenario(model_hparams['scenario'])

    # Dataset preprocessing
    train_tf_dataset = train_dataset.get_datasets(mode='train').batch(args.batch_size, drop_remainder=True)
    val_tf_dataset = val_dataset.get_datasets(mode='val').batch(args.batch_size, drop_remainder=True)

    postprocess, model_hparams = setup_image_inputs(args, scenario, train_dataset, model_hparams)

    net = model(hparams=model_hparams, batch_size=args.batch_size, scenario=scenario)
    train_tf_dataset = train_tf_dataset.shuffle(buffer_size=1024, seed=seed)
    train_tf_dataset = train_tf_dataset.prefetch(args.batch_size)
    val_tf_dataset = val_tf_dataset.prefetch(args.batch_size)

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
          epochs=args.epochs,
          loss_function=binary_classification_loss_function,
          metrics_function=binary_classification_metrics_function,
          postprocess=postprocess,
          checkpoint=args.checkpoint,
          log_path=log_path,
          log_scalars_every=args.log_scalars_every)


def eval_main(args, seed: int):
    ###############
    # Model
    ###############
    model_hparams = json.load((args.checkpoint / 'hparams.json').open('r'))
    model = link_bot_classifiers.get_model(model_hparams['model_class'])
    scenario = get_scenario(model_hparams['scenario'])
    net = model(hparams=model_hparams, batch_size=args.batch_size, scenario=scenario)

    ###############
    # Dataset
    ###############
    test_dataset = ClassifierDataset(args.dataset_dirs)
    test_tf_dataset = test_dataset.get_datasets(mode=args.mode)
    postprocess, model_hparams = setup_image_inputs(args, scenario, test_dataset, model_hparams)

    ###############
    # Evaluate
    ###############
    test_tf_dataset = test_tf_dataset.batch(args.batch_size, drop_remainder=True)
    evaluate(keras_model=net,
             test_tf_dataset=test_tf_dataset,
             loss_function=binary_classification_loss_function,
             metrics_function=binary_classification_metrics_function,
             postprocess=postprocess,
             checkpoint_path=args.checkpoint)


def main():
    np.set_printoptions(linewidth=250, precision=4, suppress=True)
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--epochs', type=int, default=15)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--log-scalars-every', type=int, help='loss/accuracy every this many steps/batches',
                              default=100)
    train_parser.add_argument('--validation-every', type=int, help='report validation every this many epochs',
                              default=1)
    train_parser.set_defaults(func=train_main)
    train_parser.add_argument('--seed', type=int, default=None)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    eval_parser.add_argument('--batch-size', type=int, default=32)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.set_defaults(func=eval_main)
    eval_parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    if args.seed is None:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    print("Using seed {}".format(seed))
    np.random.seed(seed)
    tf.random.set_random_seed(seed)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args, seed)


if __name__ == '__main__':
    main()
