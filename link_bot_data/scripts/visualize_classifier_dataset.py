#!/usr/bin/env python
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_data.image_classifier_dataset import ImageClassifierDataset
from link_bot_data.new_classifier_dataset import NewClassifierDataset
from link_bot_pycommon import link_bot_pycommon

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
    parser.add_argument("--dataset-type", choices=['image', 'new'], default='new')
    parser.add_argument('--balance-key', type=str)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    if args.dataset_type == 'image':
        classifier_dataset = ImageClassifierDataset(args.dataset_dirs)
        dataset = classifier_dataset.get_datasets(mode=args.mode,
                                                  shuffle=args.shuffle,
                                                  batch_size=1,
                                                  n_parallel_calls=1,
                                                  balance_key=None,
                                                  seed=args.seed)
    elif args.dataset_type == 'new':
        # classifier_dataset = ClassifierDataset(args.dataset_dirs)
        classifier_dataset = NewClassifierDataset(args.dataset_dirs)
        dataset = classifier_dataset.get_datasets(mode=args.mode,
                                                  shuffle=args.shuffle,
                                                  batch_size=1,
                                                  n_parallel_calls=1,
                                                  balance_key=args.balance_key,
                                                  seed=args.seed)
    done = False

    def press(event):
        nonlocal done
        if event.key == 'x':
            done = True

    fig = plt.figure()
    ax = plt.gca()
    plt.show(block=False)
    fig.canvas.mpl_connect('key_press_event', press)

    positive_count = 0
    negative_count = 0
    count = 0
    regression_with_label_1 = []
    regression_with_label_0 = []
    angles = []
    regressions = []
    for i, example in enumerate(dataset):

        if done:
            break

        label = example['label'].numpy().squeeze()
        if label:
            positive_count += 1
        else:
            negative_count += 1

        count += 1

        if args.dataset_type == 'image':
            if not args.no_plot:
                i = np.sum(example['image'].numpy(), axis=3).squeeze()
                ax.imshow(i)
                plt.draw()
                plt.title(label)
                plt.pause(2)
                plt.cla()

        elif args.dataset_type == 'new':
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

            regression = post_transition_distance - pre_transition_distance
            regressions.append(regression)
            if label:
                regression_with_label_1.append(regression)
            else:
                regression_with_label_0.append(regression)

            state_angle = link_bot_pycommon.angle_from_configuration(state)
            angles.append(state_angle)

            if not args.no_plot:
                # if label == 0:
                title = "Example {}".format(i)
                plot_classifier_data(
                    ax=ax,
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
                plt.draw()
                plt.pause(3)
                plt.cla()
            # if label == 1:
            #     plt.draw()
            #     plt.pause(0.01)
            #     plt.cla()

    class_balance = positive_count / count * 100
    print("Number of examples: {}".format(count))
    print("Class balance: {:4.1f}% positive".format(class_balance))

    print("mean median min")
    print('label 1', np.mean(regression_with_label_1), np.median(regression_with_label_1), np.min(regression_with_label_1))
    print('label 0', np.mean(regression_with_label_0), np.median(regression_with_label_0), np.min(regression_with_label_0))

    if not args.no_plot:
        plt.figure()
        plt.scatter(angles, regressions)
        plt.plot([0, np.pi], [0, 0], c='k')
        plt.xlabel("angle (rad)")
        plt.ylabel("increase in prediction error in R^{n_state} (m)")
        plt.show()


if __name__ == '__main__':
    main()
