#!/usr/bin/env python
import tensorflow as tf
import argparse
import pathlib
import matplotlib.pyplot as plt

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_planning.visualization import plot_classifier_data

tf.enable_eager_execution()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=pathlib.Path)
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no-plot', action='store_true', help='only print statistics')
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')

    args = parser.parse_args()

    classifier_dataset = ClassifierDataset(args.input_dir)
    dataset = classifier_dataset.get_dataset(mode=args.mode,
                                             shuffle=args.shuffle,
                                             num_epochs=1,
                                             batch_size=1,
                                             seed=args.seed)

    positive_count = 0
    negative_count = 0
    count = 0
    for i, example_dict in enumerate(dataset):
        planned_sdf = example_dict['planned_sdf/sdf'].numpy().squeeze()
        planned_sdf_extent = example_dict['planned_sdf/extent'].numpy().squeeze()
        actual_sdf = example_dict['actual_sdf/sdf'].numpy().squeeze()
        actual_sdf_extent = example_dict['actual_sdf/extent'].numpy().squeeze()
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
        if not args.no_plot:
            title = "Example {}".format(i)
            plot_classifier_data(
                actual_sdf=actual_sdf,
                actual_sdf_extent=actual_sdf_extent,
                next_state=next_state,
                planned_next_state=planned_next_state,
                planned_sdf=planned_sdf,
                planned_sdf_extent=planned_sdf_extent,
                planned_state=planned_state,
                state=state,
                title=title,
                label=label)
            plt.show()

    class_balance = positive_count / count * 100
    print("Number of examples: {}".format(count))
    print("Class balance: {:4.1f}% positive".format(class_balance))


if __name__ == '__main__':
    main()