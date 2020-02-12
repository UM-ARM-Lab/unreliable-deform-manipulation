#!/usr/bin/env python
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Fore

import link_bot_classifiers
from link_bot_classifiers import visualization
from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_data.image_classifier_dataset import ImageClassifierDataset
from link_bot_data.new_classifier_dataset import NewClassifierDataset
from link_bot_planning import classifier_utils
from link_bot_pycommon import experiments_util, link_bot_sdf_utils

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train(args, seed: int):
    if args.log:
        log_path = experiments_util.experiment_name(args.log)
    else:
        log_path = None

    model_hparams = json.load(args.model_hparams.open('r'))

    hparams_path = args.dataset_dirs[0] / 'hparams.json'
    dataset_hparams = json.load(hparams_path.open("r"))
    dataset_type = dataset_hparams['type']

    ###############
    # Datasets
    ###############
    if dataset_type == 'image':
        train_dataset = ImageClassifierDataset(args.dataset_dirs)
        val_dataset = ImageClassifierDataset(args.dataset_dirs)
    elif dataset_type == 'new':
        train_dataset = NewClassifierDataset(args.dataset_dirs)
        val_dataset = NewClassifierDataset(args.dataset_dirs)
    else:
        raise ValueError()

    train_tf_dataset = train_dataset.get_datasets(mode='train',
                                                  shuffle=True,
                                                  seed=seed,
                                                  batch_size=args.batch_size)
    val_tf_dataset = val_dataset.get_datasets(mode='val',
                                              shuffle=True,
                                              seed=seed,
                                              batch_size=args.batch_size)

    ###############
    # Model
    ###############
    model_hparams['classifier_dataset_hparams'] = train_dataset.hparams
    module = link_bot_classifiers.get_model_module(model_hparams['model_class'])

    try:
        ###############
        # Train
        ###############
        module.train(model_hparams, train_tf_dataset, val_tf_dataset, log_path, args, dataset_type)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "Interrupted." + Fore.RESET)
        pass


def eval(args, seed: int):
    ###############
    # Dataset
    ###############
    hparams_path = args.dataset_dirs[0] / 'hparams.json'
    dataset_hparams = json.load(hparams_path.open("r"))
    dataset_type = dataset_hparams['type']
    if dataset_type == 'image':
        test_dataset = ImageClassifierDataset(args.dataset_dirs)
    elif dataset_type == 'new':
        test_dataset = NewClassifierDataset(args.dataset_dirs)
    else:
        raise ValueError()

    test_tf_dataset = test_dataset.get_datasets(mode=args.mode,
                                                shuffle=False,
                                                seed=seed,
                                                batch_size=args.batch_size)

    ###############
    # Model
    ###############
    model_hparams = json.load((args.checkpoint / 'hparams.json').open('r'))
    module = link_bot_classifiers.get_model_module(model_hparams['model_class'])

    try:
        ###############
        # Evaluate
        ###############
        module.eval(model_hparams, test_tf_dataset, args, dataset_type)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "Interrupted." + Fore.RESET)
        pass


def eval_wrapper(args, seed: int):
    classifier_model_dir = args.checkpoint
    classifier_model = classifier_utils.load_generic_model(classifier_model_dir, 'raster')

    ###############
    # Dataset
    ###############

    hparams_path = args.dataset_dirs[0] / 'hparams.json'
    dataset_hparams = json.load(hparams_path.open("r"))
    dataset_type = dataset_hparams['type']

    ###############
    # Datasets
    ###############
    if dataset_type == 'image':
        test_dataset = ImageClassifierDataset(args.dataset_dirs)
    elif dataset_type == 'new':
        test_dataset = NewClassifierDataset(args.dataset_dirs)
    test_tf_dataset = test_dataset.get_datasets(mode=args.mode,
                                                shuffle=False,
                                                seed=seed,
                                                batch_size=1)

    for example in test_tf_dataset:
        if dataset_type == 'image':
            image = example['image']

            accept_probability = classifier_model.predict_from_image(image).squeeze()

            label = example['label'].numpy().squeeze()

            prediction = 1 if accept_probability > 0.5 else 0
            if prediction == label:
                title = 'P(accept) = {:04.3f}%, Label={}'.format(100 * accept_probability, label)
            else:
                title = 'WRONG!!! P(accept) = {:04.3f}%, Label={}'.format(100 * accept_probability, label)

            plt.figure()
            interpretable_image = visualization.make_interpretable_image(image.numpy(), classifier_model.net.n_points)
            plt.imshow(interpretable_image)
            plt.title(title)
            plt.show(block=True)
        elif dataset_type == 'new':
            local_env = example['planned_local_env/env'].numpy().squeeze()
            origin = example['planned_local_env/origin'].numpy().squeeze()
            res = example['resolution'].numpy().squeeze()
            res_2d = np.array([res, res])
            local_env_data = link_bot_sdf_utils.OccupancyData(data=local_env,
                                                              resolution=res_2d,
                                                              origin=origin)
            planned_state = example['planned_state'].numpy()
            planned_next_state = example['planned_state_next'].numpy()
            state = example['state'].numpy()
            next_state = example['state_next'].numpy()
            action = example['action'].numpy()
            label = example['label'].numpy().squeeze()

            accept_probability = classifier_model.predict([local_env_data], planned_state, planned_next_state, action)[0]

            prediction = 1 if accept_probability > 0.5 else 0
            if prediction == label:
                title = 'P(accept) = {:04.3f}%, Label={}'.format(100 * accept_probability, label)
            else:
                title = 'WRONG!!! P(accept) = {:04.3f}%, Label={}'.format(100 * accept_probability, label)

            plot_classifier_data(planned_env=local_env_data.data,
                                 planned_env_extent=local_env_data.extent,
                                 planned_state=planned_state[0],
                                 planned_next_state=planned_next_state[0],
                                 planned_env_origin=local_env_data.origin,
                                 res=local_env_data.resolution,
                                 state=state[0],
                                 next_state=next_state[0],
                                 title=title,
                                 actual_env=None,
                                 actual_env_extent=None,
                                 label=prediction)
            plt.legend()
            plt.show()


def main():
    np.set_printoptions(linewidth=250, precision=4, suppress=True)
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    train_parser.add_argument('model_hparams', type=pathlib.Path)
    train_parser.add_argument('--checkpoint', type=pathlib.Path)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--summary-freq', type=int, default=1)
    train_parser.add_argument('--save-freq', type=int, default=1)
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--log', '-l')
    train_parser.add_argument('--verbose', '-v', action='count', default=0)
    train_parser.add_argument('--log-grad-every', type=int, help='gradients hists every this many steps/batches', default=1000)
    train_parser.add_argument('--log-scalars-every', type=int, help='loss/accuracy every this many steps/batches', default=500)
    train_parser.add_argument('--validation-every', type=int, help='report validation every this many epochs', default=2000)
    train_parser.set_defaults(func=train)
    train_parser.add_argument('--seed', type=int, default=None)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_parser.add_argument('--mode', type=str, choices=['test', 'val', 'train'], default='test')
    eval_parser.add_argument('--batch-size', type=int, default=32)
    eval_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_parser.set_defaults(func=eval)
    eval_parser.add_argument('--seed', type=int, default=None)

    eval_wrapper_parser = subparsers.add_parser('evalw')
    eval_wrapper_parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    eval_wrapper_parser.add_argument('checkpoint', type=pathlib.Path)
    eval_wrapper_parser.add_argument('--mode', type=str, choices=['test', 'val', 'train'], default='test')
    eval_wrapper_parser.add_argument('--batch-size', type=int, default=32)
    eval_wrapper_parser.add_argument('--verbose', '-v', action='count', default=0)
    eval_wrapper_parser.set_defaults(func=eval_wrapper)
    eval_wrapper_parser.add_argument('--seed', type=int, default=None)

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
