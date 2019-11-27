#!/usr/bin/env python

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Style

from link_bot_classifiers.collision_checker_classifier import CollisionCheckerClassifier
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.visualization import plot_rope_configuration
from link_bot_planning.visualization import plot_classifier_data
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter

tf.compat.v1.enable_eager_execution()


def show_error(state,
               action,
               next_state,
               planned_state,
               planned_next_state,
               local_env_image,
               extent,
               prediction,
               label):
    fig, ax = plt.subplots()
    arena_size = 0.5

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim([-arena_size, arena_size])
    ax.set_ylim([-arena_size, arena_size])
    ax.axis("equal")

    ax.imshow(local_env_image, extent=extent, zorder=0, alpha=0.5)

    plot_rope_configuration(ax, state, linewidth=3, zorder=1, c='r', label='state')
    plot_rope_configuration(ax, next_state, linewidth=3, zorder=1, c='orange', label='next state')
    plot_rope_configuration(ax, planned_state, linewidth=3, zorder=2, c='b', label='planned state', linestyle='--')
    plot_rope_configuration(ax, planned_next_state, linewidth=3, zorder=2, c='c', label='planned next state', linestyle='--')
    ax.quiver(state[4], state[5], action[0], action[1], color='k')

    plt.title("prediction: {}, label: {}".format(prediction, label))
    plt.legend()
    plt.show()


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('dataset_dir', help='dataset directory', type=pathlib.Path)
    parser.add_argument('--balance', action='store_true', help='subsample the datasets to make sure it is balanced')
    parser.add_argument('--show-fn', action='store_true', help='visualize')
    parser.add_argument('--show-fp', action='store_true', help='visualize')
    parser.add_argument('--pre', type=float)
    parser.add_argument('--post', type=float)
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='test', help='mode')

    args = parser.parse_args()

    np.random.seed(0)
    tf.random.set_random_seed(0)
    balance_key = 'label' if args.balance else None

    classifier_dataset = ClassifierDataset(args.dataset_dir)
    if args.post:
        classifier_dataset.hparams['labeling']['post_close_threshold'] = args.post
    if args.pre:
        classifier_dataset.hparams['labeling']['pre_close_threshold'] = args.pre
    dataset = classifier_dataset.get_dataset(mode=args.mode,
                                             shuffle=False,
                                             seed=0,
                                             batch_size=1,
                                             balance_key=balance_key)

    collision_classifier = CollisionCheckerClassifier()

    incorrect = 0
    correct = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    neg = 0
    pos = 0
    for example_dict in dataset:
        state = example_dict['state'].numpy().squeeze()
        next_state = example_dict['state_next'].numpy().squeeze()
        planned_state = example_dict['planned_state'].numpy().squeeze()
        planned_next_state = example_dict['planned_state_next'].numpy().squeeze()
        local_env = example_dict['planned_local_env/env'].numpy().squeeze()
        action = example_dict['action'].numpy().squeeze()
        extent = example_dict['planned_local_env/extent'].numpy().squeeze()
        _res = example_dict['resolution'].numpy().squeeze()
        resolution = np.array([_res, _res])
        origin = example_dict['planned_local_env/origin'].numpy().squeeze()
        actual_local_env = example_dict['actual_local_env/env'].numpy().squeeze()
        actual_env_extent = example_dict['actual_local_env/extent'].numpy().squeeze()
        label = example_dict['label'].numpy().squeeze()

        local_env_image = np.flipud(local_env)

        if label:
            pos += 1
        else:
            neg += 1

        local_env_data = link_bot_sdf_utils.OccupancyData(local_env, resolution, origin)
        try:
            prediction = collision_classifier.predict(local_env_data,
                                                      np.expand_dims(planned_state, axis=0),
                                                      np.expand_dims(planned_next_state, axis=0))
        except IndexError:
            title = "out-of-bounds"
            plot_classifier_data(
                planned_next_state=planned_next_state,
                planned_env=local_env,
                planned_env_extent=extent,
                planned_state=planned_state,
                planned_env_origin=origin,
                res=resolution,
                state=state,
                next_state=next_state,
                title=title,
                actual_env=actual_local_env,
                actual_env_extent=actual_env_extent,
                label=label)
            plt.show()
            continue

        if prediction == 1:
            if label == 1:
                correct += 1
                tp += 1
            elif label == 0:
                incorrect += 1
                fp += 1
                if args.show_fp:
                    show_error(state, action, next_state, planned_state, planned_next_state, local_env_image, extent, prediction,
                               label)
        elif prediction == 0:
            if label == 0:
                correct += 1
                tn += 1
            elif label == 1:
                incorrect += 1
                fn += 1
                if args.show_fn:
                    show_error(state, action, next_state, planned_state, planned_next_state, local_env_image, extent,
                               prediction, label)

    print("Confusion Matrix:")
    print("|                      | label 0 | label 1 |")
    print("|     in collision (0) | {:7d} | {:7d} |".format(tn, fn))
    print("| not in collision (1) | {:7d} | {:7d} |".format(fp, tp))

    class_balance = pos / (neg + pos) * 100
    print("Class balance: {:4.1f}%".format(class_balance))

    accuracy = correct / (correct + incorrect)
    print(Style.BRIGHT + "accuracy: {:5.3f}".format(accuracy) + Style.RESET_ALL)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("precision: {:5.3f}".format(precision))
    print("recall: {:5.3f}".format(recall))


if __name__ == '__main__':
    main()
