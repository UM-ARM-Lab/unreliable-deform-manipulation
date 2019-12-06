#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import argparse
import pathlib
import matplotlib.pyplot as plt

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_classifiers.visualization import plot_classifier_data
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
    parser.add_argument('--balance-key', type=str)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)

    # classifier_dataset = ClassifierDataset(args.dataset_dirs)
    classifier_dataset = NewClassifierDataset(args.dataset_dirs)
    dataset = classifier_dataset.get_datasets(mode=args.mode,
                                              shuffle=args.shuffle,
                                              batch_size=1,
                                              n_parallel_calls=1,
                                              balance_key=args.balance_key,
                                              seed=args.seed)

    positive_count = 0
    negative_count = 0
    count = 0
    regression_with_label_1 = []
    regression_with_label_0 = []
    angles = []
    regressions = []
    for i, example in enumerate(dataset):
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
        label = example['label'].numpy().squeeze()
        pre_transition_distance = example['pre_dist'].numpy().squeeze()
        post_transition_distance = example['post_dist'].numpy().squeeze()
        if label:
            positive_count += 1
        else:
            negative_count += 1

        count += 1

        regression = post_transition_distance - pre_transition_distance
        regressions.append(regression)
        if label:
            regression_with_label_1.append(regression)
        else:
            regression_with_label_0.append(regression)

        state_angle = link_bot_pycommon.angle_from_configuration(state)
        angles.append(state_angle)

        if not args.no_plot and i < 20:
            title = "Example {}".format(i)
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
            plt.show()

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
        plt.ylabel("increase in prediction error in R6 (m)")
        plt.show()


if __name__ == '__main__':
    main()
