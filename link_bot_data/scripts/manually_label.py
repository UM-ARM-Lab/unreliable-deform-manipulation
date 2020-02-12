#!/usr/bin/env python
import argparse
import pathlib
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_data.classifier_dataset import ClassifierDataset

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf.compat.v1.enable_eager_execution(config=config)


def main():
    # THIS SHOULD BE RUN ON A DYNAMICS DATASET, not the classifier dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=pathlib.Path, nargs='+')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument("--compression-type", choices=['', 'ZLIB', 'GZIP'], default='ZLIB')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='test', help='mode')

    args = parser.parse_args()

    classifier_dataset = ClassifierDataset(args.dataset_dir)
    classifier_dataset.hparams['labeling']['discard_pre_far'] = False
    classifier_dataset.hparams['labeling']['pre_close_threshold'] = 0
    classifier_dataset.hparams['labeling']['post_close_threshold'] = 0
    dataset = classifier_dataset.get_datasets(mode=args.mode, batch_size=1, shuffle=False, seed=1)

    labeled_data = {
        'positive': [],
        'negative': [],
    }
    done = False
    t0 = None
    label = None
    for example in dataset:
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
        time_idx = example['time_idx'].numpy().squeeze()

        if label == 0:
            if time_idx != 0:
                continue

        label = (post_transition_distance < 1.0)

        manual_label = None

        fig = plt.figure()
        ax = plt.gca()

        def keypress_label(event):
            nonlocal manual_label, done
            dt = time() - t0
            if dt < 1:
                # be patient, good labeler!
                return

            if event.key == 'y':
                plt.close(fig)
                manual_label = 'positive'
            if event.key == 'n':
                plt.close(fig)
                manual_label = 'negative'
            if event.key == 'x':
                plt.close(fig)
                done = True

        fig.canvas.mpl_connect('key_press_event', keypress_label)

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
            title="t={}".format(time_idx),
            actual_env=actual_local_env,
            label=label,
            actual_env_extent=actual_local_env_extent)
        plt.legend()
        t0 = time()
        plt.show(block=True)

        if done:
            break

        if manual_label is not None:
            labeled_data[manual_label].append([pre_transition_distance, post_transition_distance])

    positive_examples = np.array(labeled_data['positive'])
    pre_positives = positive_examples[:, 0]
    post_positives = positive_examples[:, 1]
    negative_examples = np.array(labeled_data['negative'])
    pre_negatives = negative_examples[:, 0]
    post_negatives = negative_examples[:, 1]
    plt.scatter(pre_positives, post_positives, label='positive')
    plt.scatter(pre_negatives, post_negatives, label='negative')
    plt.xlabel('pre')
    plt.ylabel('post')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
