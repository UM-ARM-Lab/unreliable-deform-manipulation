#!/usr/bin/env python
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_classifiers import visualization
from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_data.image_classifier_dataset import ImageClassifierDataset
from link_bot_data.new_classifier_dataset import NewClassifierDataset
from link_bot_pycommon.link_bot_pycommon import n_state_to_n_points

tf.compat.v1.enable_eager_execution()


def main():
    np.set_printoptions(suppress=True, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no-plot', action='store_true', help='only print statistics')
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')
    parser.add_argument('--balance-key', type=str)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    # FIXME: use the same code that checks all and warns if they differ
    hparams_path = args.dataset_dirs[0] / 'hparams.json'
    dataset_hparams = json.load(hparams_path.open("r"))
    dataset_type = dataset_hparams['type']

    # FIXME: should only be one dataset class
    if dataset_type == 'image':
        classifier_dataset = ImageClassifierDataset(args.dataset_dirs)
        dataset = classifier_dataset.get_datasets(mode=args.mode,
                                                  shuffle=args.shuffle,
                                                  batch_size=1,
                                                  n_parallel_calls=1,
                                                  balance_key=None,
                                                  seed=args.seed)
    elif dataset_type == 'new':
        # classifier_dataset = ClassifierDataset(args.dataset_dirs)
        classifier_dataset = NewClassifierDataset(args.dataset_dirs)
        dataset = classifier_dataset.get_datasets(mode=args.mode,
                                                  shuffle=args.shuffle,
                                                  batch_size=1,
                                                  n_parallel_calls=1,
                                                  balance_key=args.balance_key,
                                                  seed=args.seed)
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

        if dataset_type == 'image':
            image = example['image'].numpy()
            n_points = n_state_to_n_points(dataset_hparams['n_state'])
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
        elif dataset_type == 'new':
            res = example['resolution'].numpy().squeeze()
            res = np.array([res, res])
            planned_local_env = example['planned_local_env/env'].numpy().squeeze()
            planned_local_env_extent = example['planned_local_env/extent'].numpy().squeeze()
            planned_local_env_origin = example['planned_local_env/origin'].numpy().squeeze()
            actual_local_env = example['actual_local_env/env'].numpy().squeeze()
            actual_local_env_extent = example['actual_local_env/extent'].numpy().squeeze()
            state = example['state'].numpy().squeeze()
            action = example['action'].numpy().squeeze()
            next_state = example['state_next'].numpy().squeeze()
            planned_state = example['planned_state'].numpy().squeeze()
            planned_next_state = example['planned_state_next'].numpy().squeeze()
            pre_transition_distance = example['pre_dist'].numpy().squeeze()
            post_transition_distance = example['post_dist'].numpy().squeeze()

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
