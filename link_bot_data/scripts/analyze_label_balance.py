#!/usr/bin/env python
import argparse
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.video_prediction_dataset_utils import float_feature
from link_bot_planning.visualization import plot_classifier_data

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=pathlib.Path)
    parser.add_argument('--n-examples-per-record', type=int, default=1024)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--pre', type=float, default=0.23)
    parser.add_argument('--post', type=float, default=0.23)
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='test', help='mode')
    parser.add_argument("--skip", action='store_true', help="do not write labels, only analyze")

    args = parser.parse_args()

    pre_dists = []
    post_dists = []
    positive_labels = 0
    negative_labels = 0

    n_both_close = 0
    n_both_far = 0
    n_pre_close_post_far = 0
    n_pre_far_post_close = 0

    classifier_dataset = ClassifierDataset(args.indir)
    dataset = classifier_dataset.get_dataset(mode=args.mode, num_epochs=1, batch_size=0, shuffle=False)

    for example_dict in dataset:
        state = example_dict['state'].numpy()
        next_state = example_dict['next_state'].numpy()
        planned_state = example_dict['planned_state'].numpy()
        planned_next_state = example_dict['planned_next_state'].numpy()

        # for visualization only
        res = example_dict['res'].numpy().squeeze()
        res = np.array([res, res])
        planned_local_env = example_dict['planned_local_env/env'].numpy().squeeze()
        planned_local_env_extent = example_dict['planned_local_env/extent'].numpy().squeeze()
        planned_local_env_origin = example_dict['planned_local_env/origin'].numpy().squeeze()
        actual_local_env = example_dict['actual_local_env/env'].numpy().squeeze()
        actual_local_env_extent = example_dict['actual_local_env/extent'].numpy().squeeze()
        label = example_dict['label'].numpy().squeeze()

        # Compute the label for whether our model should be trusted
        pre_transition_distance = np.linalg.norm(state - planned_state)
        post_transition_distance = np.linalg.norm(next_state - planned_next_state)

        pre_dists.append(pre_transition_distance)
        post_dists.append(post_transition_distance)

        pre_close = pre_transition_distance < args.pre
        post_close = post_transition_distance < args.post

        if pre_close and post_close:
            positive_labels += 1
        else:
            negative_labels += 1

        if pre_close:
            if post_close:
                n_both_close += 1
            elif not post_close:
                n_pre_close_post_far += 1

                # plot_classifier_data(
                #     actual_env=actual_local_env,
                #     actual_env_extent=actual_local_env_extent,
                #     next_state=next_state,
                #     planned_next_state=planned_next_state,
                #     planned_env=planned_local_env,
                #     planned_env_extent=planned_local_env_extent,
                #     planned_state=planned_state,
                #     planned_env_origin=planned_local_env_origin,
                #     res=res,
                #     state=state,
                #     title='',
                #     label=label)
                # plt.show()
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
