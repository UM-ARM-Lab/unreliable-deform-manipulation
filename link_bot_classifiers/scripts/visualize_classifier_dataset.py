#!/usr/bin/env python
import argparse
import json
import time
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import link_bot_classifiers
from link_bot_classifiers import visualization
from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_data.classifier_dataset import ClassifierDataset, convert_sequences_to_transitions
from link_bot_data.link_bot_dataset_utils import balance_by_augmentation
from link_bot_data.visualization import plot_rope_configuration
from link_bot_pycommon.link_bot_pycommon import n_state_to_n_points, print_dict

tf.compat.v1.enable_eager_execution()


def main():
    np.set_printoptions(suppress=True, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('classifier_dataset_params', type=pathlib.Path)
    parser.add_argument('model_hparams', type=pathlib.Path)
    parser.add_argument('display_type', choices=['transition_image', 'transition_plot', 'trajectory_plot'])
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pre', type=int, default=0.15)
    parser.add_argument('--post', type=int, default=0.21)
    parser.add_argument('--discard-pre-far', action='store_true')
    parser.add_argument('--no-plot', action='store_true', help='only print statistics')
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.compat.v1.random.set_random_seed(args.seed)

    classifier_dataset_params = json.load(args.classifier_dataset_params.open("r"))

    classifier_dataset = ClassifierDataset(args.dataset_dirs, classifier_dataset_params)
    dataset = classifier_dataset.get_datasets(mode=args.mode,
                                              shuffle=args.shuffle,
                                              batch_size=1,
                                              n_parallel_calls=1,
                                              seed=args.seed)

    model_hparams = json.load(args.model_hparams.open('r'))
    model_hparams['classifier_dataset_hparams'] = classifier_dataset.hparams
    module = link_bot_classifiers.get_model_module(model_hparams['model_class'])

    net = module.model(model_hparams, batch_size=1)
    dataset = net.post_process(dataset)

    # if classifier_dataset_params['balance']:
        # TODO: make this faster somehow
        # dataset = balance_by_augmentation(dataset, key='label')

    done = False

    positive_count = 0
    negative_count = 0
    count = 0
    for i, example in enumerate(dataset):

        if done:
            break

        label = example['label'].numpy().squeeze()
        if label:
            positive_count += 1
        else:
            negative_count += 1

        count += 1

        if args.no_plot:
            continue

        if args.display_type == 'transition_image':
            image = example['image'].numpy()
            n_points = n_state_to_n_points(classifier_dataset.hparams['n_state'])
            interpretable_image = visualization.make_interpretable_image(image, n_points)
            plt.imshow(interpretable_image)
            planned_env_extent = [1, 49, 1, 49]
            label_color = 'g' if label else 'r'
            plt.plot([planned_env_extent[0], planned_env_extent[0], planned_env_extent[1], planned_env_extent[1],
                      planned_env_extent[0]],
                     [planned_env_extent[2], planned_env_extent[3], planned_env_extent[3], planned_env_extent[2],
                      planned_env_extent[2]],
                     c=label_color, linewidth=4)
            plt.show(block=True)
        elif args.display_type == 'trajectory_plot':
            full_env = example['full_env/env'].numpy()[0]
            full_env_extent = example['full_env/extent'].numpy()[0]
            link_bot_state_all = example['planned_state/link_bot_all'].numpy()[0]
            link_bot_state_stop_idx = example['planned_state/link_bot_all_stop'].numpy()[0]

            plt.figure()
            plt.imshow(full_env, extent=full_env_extent)
            ax = plt.gca()
            for i in range(link_bot_state_stop_idx):
                state = link_bot_state_all[i]
                print(state)
                plot_rope_configuration(ax, state, c='r', s=5)
            plt.show()
        elif args.display_type == 'transition_plot':
            res = example['res'].numpy().squeeze()
            res = np.array([res, res])
            planned_local_env = example['planned_state/local_env'].numpy().squeeze()
            planned_local_env_extent = example['planned_state/local_env_extent'].numpy().squeeze()
            planned_local_env_origin = example['planned_state/local_env_origin'].numpy().squeeze()
            actual_local_env = example['state/local_env'].numpy().squeeze()
            actual_local_env_extent = example['state/local_env_extent'].numpy().squeeze()
            state = example['state/link_bot'].numpy().squeeze()
            action = example['action'].numpy().squeeze()
            next_state = example['state_next/link_bot'].numpy().squeeze()
            planned_state = example['planned_state/link_bot'].numpy().squeeze()
            planned_next_state = example['planned_state_next/link_bot'].numpy().squeeze()
            pre_transition_distance = example['pre_dist'].numpy().squeeze()
            post_transition_distance = example['post_dist'].numpy().squeeze()

            title = None
            if post_transition_distance > 0.25:
                title = "BAD Example {}, {:0.3f}".format(i, post_transition_distance)
            if post_transition_distance < 0.05:
                title = "GOOD Example {}, {:0.3f}".format(i, post_transition_distance)
            plot_classifier_data(
                next_state=next_state,
                action=action,
                planned_next_state=planned_next_state,
                planned_env=planned_local_env,
                planned_env_extent=planned_local_env_extent,
                planned_state=planned_state,
                planned_env_origin=planned_local_env_origin,
                res=res,
                state=state,
                title=title,
                actual_env=actual_local_env,
                actual_env_extent=actual_local_env_extent,
                label=label)
            plt.legend()
            plt.show(block=True)

    class_balance = positive_count / count * 100
    print("Number of examples: {}".format(count))
    print("Class balance: {:4.1f}% positive".format(class_balance))


if __name__ == '__main__':
    main()
