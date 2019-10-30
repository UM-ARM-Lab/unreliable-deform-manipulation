#!/usr/bin/env python

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_classifiers.collision_checker_classifier import CollisionCheckerClassifier
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.visualization import plot_rope_configuration
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.args import my_formatter

tf.enable_eager_execution()


def show_error(state,
               next_state,
               planned_state,
               planned_next_state,
               action,
               local_env_image,
               extent,
               predicted_violated,
               label_model_reliable):
    fig, ax = plt.subplots()
    arena_size = 0.5

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim([-arena_size, arena_size])
    ax.set_ylim([-arena_size, arena_size])
    ax.axis("equal")

    ax.imshow(local_env_image, extent=extent, zorder=0, alpha=0.5)

    # plt.quiver(planned_state[4], planned_state[5], action[0], action[1], zorder=4)

    plot_rope_configuration(ax, state, linewidth=3, zorder=1, c='r', label='state')
    plot_rope_configuration(ax, next_state, linewidth=3, zorder=1, c='orange', label='next state')
    plot_rope_configuration(ax, planned_state, linewidth=3, zorder=1, c='b', label='planned state')
    plot_rope_configuration(ax, planned_next_state, linewidth=3, zorder=1, c='c', label='planned next state')

    plt.scatter(planned_state[4], planned_state[5], c='r' if predicted_violated else 'g', s=200, zorder=2, label='pred')
    plt.scatter(planned_state[4], planned_state[5], c='g' if label_model_reliable else 'r', marker='*', s=150, zorder=3,
                linewidths=0.1,
                edgecolors='k', label='true')
    plt.title("speed ({:.3f},{:.3f})m/s".format(action[0], action[1]))
    plt.legend()
    plt.show()


def main():
    np.set_printoptions(suppress=True, linewidth=250, precision=4, threshold=1000)
    tf.logging.set_verbosity(tf.logging.FATAL)

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument('input_dir', help='dataset directory', type=pathlib.Path)
    parser.add_argument('--dataset-hparams-dict', help='override dataset hyperparams')
    parser.add_argument('--balance', action='store_true', help='subsample the datasets to make sure it is balanced')
    parser.add_argument('--show-fn', action='store_true', help='visualize')
    parser.add_argument('--show-fp', action='store_true', help='visualize')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='test', help='mode')
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')

    args = parser.parse_args()

    np.random.seed(0)
    tf.random.set_random_seed(0)

    classifier_dataset = ClassifierDataset(args.input_dir, is_labeled=True)
    dataset = classifier_dataset.get_dataset(mode=args.mode,
                                             shuffle=False,
                                             num_epochs=1,
                                             seed=0,
                                             batch_size=None,  # nobatching
                                             )

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
        next_state = example_dict['next_state'].numpy().squeeze()
        planned_state = example_dict['planned_state'].numpy().squeeze()
        planned_next_state = example_dict['planned_next_state'].numpy().squeeze()
        local_env = example_dict['planned_local_env/env'].numpy().squeeze()
        extent = example_dict['planned_local_env/extent'].numpy().squeeze()
        res = example_dict['res'].numpy().squeeze()
        resolution = np.array([res, res])
        origin = example_dict['planned_local_env/origin'].numpy().squeeze()
        action = example_dict['action'].numpy().squeeze()
        label = example_dict['label'].numpy().squeeze()

        local_env_image = np.flipud(local_env)

        if label:
            pos += 1
        else:
            neg += 1

        local_env_data = link_bot_sdf_utils.OccupancyData(local_env, resolution, origin)
        prediction = collision_classifier.predict(local_env_data, planned_state, planned_next_state)
        if prediction == 1:
            if label == 1:
                correct += 1
                tp += 1
            elif label == 0:
                incorrect += 1
                fp += 1
                if args.show_fp:
                    show_error(state, next_state, planned_state, planned_next_state, action, local_env_image, extent, prediction,
                               label)
        elif prediction == 0:
            if label == 0:
                correct += 1
                tn += 1
            elif label == 1:
                incorrect += 1
                fn += 1
                if args.show_fn:
                    show_error(state, next_state, planned_state, planned_next_state, action, local_env_image, extent,
                               prediction, label)

    print("Confusion Matrix:")
    print("|                      | label 0 | label 1 |")
    print("|     in collision (0) | {:7d} | {:7d} |".format(tn, fn))
    print("| not in collision (1) | {:7d} | {:7d} |".format(fp, tp))

    class_balance = pos / (neg + pos) * 100
    print("Class balance: {:4.1f}%".format(class_balance))

    accuracy = correct / (correct + incorrect)
    print("accuracy: {:5.3f}".format(accuracy))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("precision: {:5.3f}".format(precision))
    print("recall: {:5.3f}".format(recall))


if __name__ == '__main__':
    main()
