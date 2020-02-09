#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_data.classifier_dataset import ClassifierDataset

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


def main():
    # THIS SHOULD BE RUN ON A DYNAMICS DATASET, not the classifier dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, nargs='+')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--pre', type=float, default=0.15)
    parser.add_argument('--post', type=float, default=0.21)
    parser.add_argument('--discard-pre-far', action='store_true')
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
    classifier_dataset.hparams['labeling']['discard_pre_far'] = args.discard_pre_far
    classifier_dataset.hparams['labeling']['pre_close_threshold'] = args.pre
    classifier_dataset.hparams['labeling']['post_close_threshold'] = args.post
    dataset = classifier_dataset.get_datasets(mode=args.mode, batch_size=1, shuffle=False, seed=1)

    speeds = []
    errors = []
    lengths = []
    for example_dict in dataset:
        state = example_dict['state'].numpy()
        next_state = example_dict['state_next'].numpy()
        action = example_dict['action'].numpy()
        planned_state = example_dict['planned_state'].numpy()
        planned_next_state = example_dict['planned_state_next'].numpy()
        label = example_dict['label'].numpy()

        speed = np.linalg.norm(action)
        error = np.linalg.norm(state - planned_state)
        points = planned_state.reshape(-1, 2)
        deltas = points[1:] - points[:-1]
        distances = np.linalg.norm(deltas, axis=1)
        rope_length = np.sum(distances)
        if speed > 0 and error > 0:
            lengths.append(rope_length)
            speeds.append(speed)
            errors.append(error)

        # Compute the label for whether our model should be trusted
        pre_transition_distance = np.linalg.norm(state - planned_state)
        post_transition_distance = np.linalg.norm(next_state - planned_next_state)

        pre_dists.append(pre_transition_distance)
        post_dists.append(post_transition_distance)

        pre_close = pre_transition_distance < args.pre
        post_close = post_transition_distance < args.post

        if pre_close and post_close:
            n_both_close += 1
        elif pre_close and not post_close:
            n_pre_close_post_far += 1
        elif not pre_close and post_close:
            n_pre_far_post_close += 1
        else:
            n_both_far += 1

        if args.discard_pre_far and not pre_close:
            continue

        if post_close:
            positive_labels += 1
        else:
            negative_labels += 1

    plt.figure()
    plt.title("accuracy vs speed")
    plt.scatter(speeds, errors)
    plt.xlabel("speed m/s")
    plt.ylabel("error")

    plt.figure()
    plt.title("accuracy vs length")
    plt.scatter(lengths, errors)
    plt.xlabel("length m")
    plt.ylabel("error")
    plt.show(block=True)

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
