#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import argparse
import pathlib
import matplotlib.pyplot as plt

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_planning.visualization import plot_classifier_data

tf.enable_eager_execution()


def main():
    np.set_printoptions(suppress=True, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path)
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no-plot', action='store_true', help='only print statistics')
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')

    args = parser.parse_args()

    classifier_dataset = ClassifierDataset(args.input_dir)
    dataset = classifier_dataset.cf_get_dataset(mode=args.mode,
                                                shuffle=args.shuffle,
                                                batch_size=1,
                                                seed=args.seed)

    positive_count = 0
    negative_count = 0
    count = 0
    regression_with_label_1 = []
    regression_with_label_0 = []
    for i, example_dict in enumerate(dataset):
        res = example_dict['res'].numpy().squeeze()
        res = np.array([res, res])
        planned_local_env = example_dict['planned_local_env/env'].numpy().squeeze()
        planned_local_env_extent = example_dict['planned_local_env/extent'].numpy().squeeze()
        planned_local_env_origin = example_dict['planned_local_env/origin'].numpy().squeeze()
        actual_local_env = example_dict['actual_local_env/env'].numpy().squeeze()
        actual_local_env_extent = example_dict['actual_local_env/extent'].numpy().squeeze()
        state = example_dict['state'].numpy().squeeze()
        next_state = example_dict['next_state'].numpy().squeeze()
        planned_state = example_dict['planned_state'].numpy().squeeze()
        planned_next_state = example_dict['planned_next_state'].numpy().squeeze()
        if classifier_dataset.is_labeled:
            label = example_dict['label'].numpy().squeeze()
            if label:
                positive_count += 1
            else:
                negative_count += 1
        else:
            label = None

        count += 1

        pre_transition_distance = np.linalg.norm(state - planned_state)
        post_transition_distance = np.linalg.norm(next_state - planned_next_state)

        regression = post_transition_distance - pre_transition_distance
        if label:
            regression_with_label_1.append(regression)
        else:
            regression_with_label_0.append(regression)

        if not args.no_plot:
            title = "Example {}".format(i)
            plot_classifier_data(
                actual_env=actual_local_env,
                actual_env_extent=actual_local_env_extent,
                next_state=next_state,
                planned_next_state=planned_next_state,
                planned_env=planned_local_env,
                planned_env_extent=planned_local_env_extent,
                planned_state=planned_state,
                planned_env_origin=planned_local_env_origin,
                res=res,
                state=state,
                title=title,
                label=label)
            plt.show()

    class_balance = positive_count / count * 100
    print("Number of examples: {}".format(count))
    print("Class balance: {:4.1f}% positive".format(class_balance))

    print("mean median min")
    print('label 1', np.mean(regression_with_label_1), np.median(regression_with_label_1), np.min(regression_with_label_1))
    print('label 0', np.mean(regression_with_label_0), np.median(regression_with_label_0), np.min(regression_with_label_0))


if __name__ == '__main__':
    main()
