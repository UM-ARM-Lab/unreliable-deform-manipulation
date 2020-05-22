#!/usr/bin/env python
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import progressbar
import tensorflow as tf
from colorama import Fore

import link_bot_classifiers
from link_bot_classifiers.visualization import visualize_classifier_example, classifier_example_title
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_pycommon.get_scenario import get_scenario
from moonshine import experiments_util
from moonshine.classifier_losses_and_metrics import binary_classification_loss_function, binary_classification_metrics_function, \
    binary_classification_sequence_loss_function, binary_classification_sequence_metrics_function
from moonshine.gpu_config import limit_gpu_mem
from moonshine.metric import AccuracyMetric
from moonshine.moonshine_utils import remove_batch
from moonshine.tensorflow_train_test_loop import evaluate, train

limit_gpu_mem(3)


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
    train_tf_dataset = train_dataset.get_datasets(mode='train', take=args.take)
    val_tf_dataset = val_dataset.get_datasets(mode='val')

    # to mix up examples so each batch is diverse
    train_tf_dataset = train_tf_dataset.shuffle(buffer_size=2048, seed=seed, reshuffle_each_iteration=True)

    train_tf_dataset = train_tf_dataset.batch(args.batch_size, drop_remainder=True)
    val_tf_dataset = val_tf_dataset.batch(args.batch_size, drop_remainder=True)

    train_tf_dataset = train_tf_dataset.shuffle(buffer_size=512, seed=seed, reshuffle_each_iteration=True)  # to mix up batches

    train_tf_dataset = train_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_tf_dataset = val_tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    ###############
    # Train
    ###############
    if model_hparams['supervize_full_sequence']:
        loss_function = binary_classification_sequence_loss_function
        metrics_function = binary_classification_sequence_metrics_function
    else:
        metrics_function = binary_classification_metrics_function
        loss_function = binary_classification_loss_function

    net = model(hparams=model_hparams, batch_size=args.batch_size, scenario=scenario)
    train(keras_model=net,
          model_hparams=model_hparams,
          train_tf_dataset=train_tf_dataset,
          val_tf_dataset=val_tf_dataset,
          dataset_dirs=args.dataset_dirs,
          seed=seed,
          batch_size=args.batch_size,
          epochs=args.epochs,
          loss_function=loss_function,
          metrics_function=metrics_function,
          checkpoint=args.checkpoint,
          key_metric=AccuracyMetric,
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

    ###############
    # Evaluate
    ###############
    test_tf_dataset = test_tf_dataset.batch(args.batch_size, drop_remainder=True)
    if model_hparams['supervize_full_sequence']:
        loss_function = binary_classification_sequence_loss_function
        metrics_function = binary_classification_sequence_metrics_function
    else:
        metrics_function = binary_classification_metrics_function
        loss_function = binary_classification_loss_function
    evaluate(keras_model=net,
             test_tf_dataset=test_tf_dataset,
             loss_function=loss_function,
             metrics_function=metrics_function,
             checkpoint_path=args.checkpoint)


def viz_main(args, seed: int):
    ###############
    # Model
    ###############
    plt.style.use("slides")
    model_hparams = json.load((args.checkpoint / 'hparams.json').open('r'))
    model = link_bot_classifiers.get_model(model_hparams['model_class'])
    scenario = get_scenario(model_hparams['scenario'])
    keras_model = model(hparams=model_hparams, batch_size=args.batch_size, scenario=scenario)

    ###############
    # Dataset
    ###############
    dataset_name = "_and_".join([d.name for d in args.dataset_dirs])
    classifier_dataset = ClassifierDataset(args.dataset_dirs, load_true_states=True)
    tf_dataset = classifier_dataset.get_datasets(mode=args.mode).batch(args.batch_size).shuffle(buffer_size=1024, seed=seed)

    ###############
    # Evaluate
    ###############
    ckpt = tf.train.Checkpoint(net=keras_model)
    manager = tf.train.CheckpointManager(ckpt, args.checkpoint, max_to_keep=1)  # doesn't matter here, we're not saving
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    print(Fore.CYAN + "Restored from {}".format(manager.latest_checkpoint) + Fore.RESET)

    outdir = args.checkpoint / f'visualizations_{dataset_name}'
    try:
        for example_idx, example in enumerate(tf_dataset):
            outputs = keras_model(example, training=False)
            accept_probabilities = outputs['probabilities']
            example = remove_batch(example)
            accept_probabilities = remove_batch(accept_probabilities).numpy().squeeze()
            last_valid_idx = int(example['last_valid_idx'].numpy().squeeze())
            print(accept_probabilities[:last_valid_idx])  # no +1 here because there's no accept probability for t=0
            accept = accept_probabilities[-1] > args.classifier_threshold

            label = example['label'].numpy().squeeze()

            false_negative = label and not accept
            false_positive = accept and not label

            if args.only_negative and label != 0:
                continue
            if args.only_positive and label != 1:
                continue
            if args.only_false_negatives and not false_negative:
                continue
            if args.only_false_positives and not false_positive:
                continue

            title = classifier_example_title(example)
            handle = visualize_classifier_example(args=args,
                                                  scenario=scenario,
                                                  outdir=outdir,
                                                  model_hparams=model_hparams,
                                                  classifier_dataset=classifier_dataset,
                                                  example=example,
                                                  example_idx=example_idx,
                                                  title=title,
                                                  accept_probabilities=accept_probabilities)
            plt.show(block=True)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "Interrupted." + Fore.RESET)


def main():
    np.set_printoptions(linewidth=250, precision=4, suppress=True, threshold=10000)
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--take', type=int)
    train_parser.add_argument('--epochs', type=int, default=150)
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
    eval_parser.add_argument('--batch-size', type=int, default=64)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.set_defaults(func=eval_main)
    eval_parser.add_argument('--seed', type=int, default=None)

    viz_parser = subparsers.add_parser('viz')
    viz_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    viz_parser.add_argument('checkpoint', type=pathlib.Path)
    viz_parser.add_argument('display_type', choices=['just_count', 'image', 'anim', 'plot'])
    viz_parser.add_argument('--batch-size', type=int, default=1)
    viz_parser.add_argument('--classifier-threshold', type=float, default=0.5)
    viz_parser.add_argument('--shuffle', action='store_true')
    viz_parser.add_argument('--no-balance', action='store_true')
    viz_parser.add_argument('--only-negative', action='store_true')
    viz_parser.add_argument('--only-positive', action='store_true')
    viz_parser.add_argument('--only-false-positives', action='store_true')
    viz_parser.add_argument('--only-false-negatives', action='store_true')
    viz_parser.add_argument('--only-reconverging', action='store_true')
    viz_parser.add_argument('--save', action='store_true')
    viz_parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test')
    viz_parser.add_argument('--verbose', '-v', action='count', default=0)
    viz_parser.set_defaults(func=viz_main)
    viz_parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    if args.seed is None:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    print("Using seed {}".format(seed))
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if args == argparse.Namespace():
        parser.print_usage()
    else:
        args.func(args, seed)


if __name__ == '__main__':
    main()
