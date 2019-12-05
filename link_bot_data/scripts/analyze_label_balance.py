#!/usr/bin/env python
import argparse
import pathlib

import numpy as np
import tensorflow as tf

from link_bot_data.classifier_dataset import ClassifierDataset

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('--n-examples-per-record', type=int, default=1024)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='test', help='mode')

    args = parser.parse_args()

    pre_dists = []
    post_dists = []
    positive_labels = 0
    negative_labels = 0

    n_both_close = 0
    n_both_far = 0
    n_pre_close_post_far = 0
    n_pre_far_post_close = 0

    classifier_dataset = ClassifierDataset(args.dataset_dir)
    dataset = classifier_dataset.get_datasets(mode=args.mode, batch_size=1, shuffle=False, seed=1)

    for example_dict in dataset:
        state = example_dict['state'].numpy()
        next_state = example_dict['state_next'].numpy()
        planned_state = example_dict['planned_state'].numpy()
        planned_next_state = example_dict['planned_state_next'].numpy()
        label = example_dict['label'].numpy()

        # Compute the label for whether our model should be trusted
        pre_transition_distance = np.linalg.norm(state - planned_state)
        post_transition_distance = np.linalg.norm(next_state - planned_next_state)

        pre_dists.append(pre_transition_distance)
        post_dists.append(post_transition_distance)

        pre_close = pre_transition_distance < args.threshold
        post_close = post_transition_distance < args.threshold

        if pre_close and post_close:
            positive_labels += 1
        else:
            negative_labels += 1

        if pre_close:
            if post_close:
                n_both_close += 1
            elif not post_close:
                n_pre_close_post_far += 1

        elif not pre_close:
            if post_close:
                n_pre_far_post_close += 1
            elif not post_close:
                n_both_far += 1

    print("Confusion Matrix:")
    print("|            | pre close | pre_far   |")
    print("| post close | {:9d} | {:9d} |".format(n_both_close, n_pre_far_post_close))
    print("|   post far | {:9d} | {:9d} |".format(n_pre_close_post_far, n_both_far))

    print("Positive labels: {}".format(positive_labels))
    print("Negative labels: {}".format(negative_labels))
    percent_positive = positive_labels / (positive_labels + negative_labels) * 100
    print("Class balance : {:3.2f}% positive".format(percent_positive))


if __name__ == '__main__':
    main()
