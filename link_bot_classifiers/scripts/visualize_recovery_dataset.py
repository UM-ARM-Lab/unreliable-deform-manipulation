#!/usr/bin/env python
from progressbar import progressbar
import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import pathlib
import rospy

from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import listify
from moonshine.gpu_config import limit_gpu_mem
from visualization_msgs.msg import MarkerArray, Marker
from link_bot_data.visualization import rviz_arrow
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import log_scale_0_to_1
from link_bot_data.recovery_dataset import RecoveryDataset


limit_gpu_mem(1)


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=5)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('--type', choices=['best_and_worst', 'in_order', 'stats'], default='best_and_worst')
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='train')

    args = parser.parse_args()

    rospy.init_node('vis_recovery_dataset')

    dataset = RecoveryDataset(args.dataset_dirs)

    if args.type == 'best_and_worst':
        visualize_best_and_worst(args, dataset)
    elif args.type == 'in_order':
        visualize_in_order(args, dataset)
    elif args.type == 'stats':
        stats(args, dataset)


def stats(args, dataset):
    recovery_probabilities = []
    batch_size = 512
    tf_dataset = dataset.get_datasets(mode=args.mode).batch(batch_size, drop_remainder=True)
    for example in tf_dataset:
        recovery_probabilities.append(tf.reduce_mean(example['recovery_probability'][:, 1]))

    overall_recovery_probability_mean = tf.reduce_mean(recovery_probabilities)

    print(f'mean recovery probability of dataset: {overall_recovery_probability_mean:.5f}')

    losses = []
    for example in tf_dataset:
        y_true = tf.reshape(example['recovery_probability'][:, 1], [batch_size, 1])
        pred = tf.reshape([overall_recovery_probability_mean] * batch_size, [batch_size, 1])
        loss = tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=pred, from_logits=False)
        losses.append(loss)
    print(f"loss to beat {tf.reduce_mean(losses)}")


def visualize_best_and_worst(args, dataset: RecoveryDataset):
    tf_dataset = dataset.get_datasets(mode=args.mode)

    # sort the dataset
    examples_to_sort = []
    for example in progressbar(tf_dataset):
        recovery_probability_1 = example['recovery_probability'][1]
        if recovery_probability_1 > 0.0:
            examples_to_sort.append(example)

    examples_to_sort = sorted(examples_to_sort, key=lambda e: e['recovery_probability'][1], reverse=True)

    # print("BEST")
    for example in examples_to_sort:
        visualize_example(dataset, example)

    # print("WORST")
    # for example in examples_to_sort[:10]:
    #     visualize_example(dataset, example)


def visualize_in_order(args, dataset: RecoveryDataset):
    tf_dataset = dataset.get_datasets(mode=args.mode)

    for example in tf_dataset:
        visualize_example(dataset, example)


def visualize_example(dataset, example):
    scenario = get_scenario(dataset.hparams['scenario'])

    anim = RvizAnimationController(np.arange(dataset.horizon))
    scenario.plot_environment_rviz(example)
    while not anim.done:

        t = anim.t()
        recovery_probability_t = example['recovery_probability'][t]
        scenario.plot_recovery_probability(recovery_probability_t)
        s_t = {k: example[k][t] for k in dataset.state_keys}
        if t < dataset.horizon - 1:
            a_t = {k: example[k][t] for k in dataset.action_keys}

            scenario.plot_action_rviz(s_t, a_t, label='observed')
        color_factor = log_scale_0_to_1(recovery_probability_t, k=10)
        scenario.plot_state_rviz(s_t, label='observed', color=cm.Reds(color_factor))
        anim.step()


if __name__ == '__main__':
    main()
