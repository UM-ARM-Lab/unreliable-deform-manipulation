#!/usr/bin/env python
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import tensorflow as tf

from link_bot_classifiers import visualization
from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import balance_by_augmentation, add_traj_image, add_transition_image
from link_bot_data.visualization import plot_rope_configuration
from link_bot_pycommon.link_bot_pycommon import n_state_to_n_points

tf.compat.v1.enable_eager_execution()


def main():
    np.set_printoptions(suppress=True, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('classifier_dataset_params', type=pathlib.Path)
    parser.add_argument('display_type', choices=['transition_image', 'transition_plot', 'trajectory_image', 'trajectory_plot'])
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pre', type=int, default=0.15)
    parser.add_argument('--post', type=int, default=0.21)
    parser.add_argument('--discard-pre-far', action='store_true')
    parser.add_argument('--action-in-image', action='store_true')
    parser.add_argument('--no-balance', action='store_true')
    parser.add_argument('--only-negative', action='store_true')
    parser.add_argument('--no-plot', action='store_true', help='only print statistics')
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.compat.v1.random.set_random_seed(args.seed)

    classifier_dataset_params = json.load(args.classifier_dataset_params.open("r"))

    classifier_dataset = ClassifierDataset(args.dataset_dirs, classifier_dataset_params)
    dataset = classifier_dataset.get_datasets(mode=args.mode)
    if args.display_type == 'transition_image':
        dataset = add_transition_image(dataset, args.action_in_image)
    if args.display_type == 'trajectory_image':
        dataset = add_traj_image(dataset, args.action_in_image)

    if classifier_dataset_params['balance'] and not args.no_balance:
        if args.display_type == 'transition_image':
            dataset = balance_by_augmentation(dataset, image_key='transition_image')
        elif args.display_type == 'trajectory_image':
            dataset = balance_by_augmentation(dataset, image_key='trajectory_image')
        else:
            print("no images, can't balance...")

    if args.shuffle:
        dataset = dataset.shuffle(buffer_size=1024)

    done = False

    positive_count = 0
    negative_count = 0
    count = 0
    for i, example in enumerate(dataset):

        if done:
            break

        label = example['label'].numpy().squeeze()

        if args.only_negative and label != 0:
            continue

        if label:
            positive_count += 1
        else:
            negative_count += 1

        count += 1

        if args.no_plot:
            continue

        if args.display_type == 'transition_image':
            image = example['transition_image'].numpy()
            next_state = example['planned_state_next/link_bot'].numpy()
            n_points = n_state_to_n_points(classifier_dataset.hparams['n_state'])
            interpretable_image = visualization.make_interpretable_image(image, n_points)
            plt.imshow(np.flipud(interpretable_image))
            planned_env_extent = [1, 49, 1, 49]
            label_color = 'g' if label else 'r'
            plt.plot([planned_env_extent[0], planned_env_extent[0], planned_env_extent[1], planned_env_extent[1],
                      planned_env_extent[0]],
                     [planned_env_extent[2], planned_env_extent[3], planned_env_extent[3], planned_env_extent[2],
                      planned_env_extent[2]],
                     c=label_color, linewidth=4)
            plt.title(next_state)
            plt.show(block=True)
        elif args.display_type == 'trajectory_image':
            image = example['trajectory_image'].numpy()
            image = np.pad(image, [[0, 0], [0, 0], [0, 1]])
            plt.imshow(np.flipud(image))
            planned_env_extent = [1, 199, 1, 199]
            label_color = 'g' if label else 'r'
            plt.plot([planned_env_extent[0], planned_env_extent[0], planned_env_extent[1], planned_env_extent[1],
                      planned_env_extent[0]],
                     [planned_env_extent[2], planned_env_extent[3], planned_env_extent[3], planned_env_extent[2],
                      planned_env_extent[2]],
                     c=label_color, linewidth=4)
            plt.show(block=True)
        elif args.display_type == 'trajectory_plot':
            full_env = example['full_env/env'].numpy()
            full_env_extent = example['full_env/extent'].numpy()
            link_bot_state_all = example['planned_state/link_bot_all'].numpy()
            link_bot_state_stop_idx = example['planned_state/link_bot_all_stop'].numpy()
            actual_link_bot_state_all = example['state/link_bot_all'].numpy()

            plt.figure()
            plt.imshow(np.flipud(full_env), extent=full_env_extent)
            ax = plt.gca()
            for i in range(link_bot_state_stop_idx):
                actual_state = actual_link_bot_state_all[i]
                planned_state = link_bot_state_all[i]
                plot_rope_configuration(ax, actual_state, c='white', s=8)
                plot_rope_configuration(ax, planned_state, c='orange', s=6)
            plt.show()
        elif args.display_type == 'transition_plot':
            res = example['res'].numpy()
            res = np.array([res, res])
            planned_local_env = example['planned_state/local_env'].numpy()
            planned_local_env_extent = example['planned_state/local_env_extent'].numpy()
            planned_local_env_origin = example['planned_state/local_env_origin'].numpy()
            actual_local_env = example['state/local_env'].numpy()
            actual_local_env_extent = example['state/local_env_extent'].numpy()
            state = example['state/link_bot'].numpy()
            action = example['action'].numpy()
            next_state = example['state_next/link_bot'].numpy()
            planned_state = example['planned_state/link_bot'].numpy()
            planned_next_state = example['planned_state_next/link_bot'].numpy()
            pre_transition_distance = example['pre_dist'].numpy()
            post_transition_distance = example['post_dist'].numpy()

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
            ax = plt.gca()
            circle = patches.Circle([0, 0], radius=2.4, edgecolor='g', facecolor='#ffffff00', linewidth=3, zorder=1)
            ax.add_patch(circle)
            plt.legend()
            plt.show(block=True)

    class_balance = positive_count / count * 100
    print("Number of examples: {}".format(count))
    print("Number positive: {}".format(positive_count))
    print("Number negative: {}".format(negative_count))
    print("Class balance: {:4.1f}% positive".format(class_balance))


if __name__ == '__main__':
    main()
