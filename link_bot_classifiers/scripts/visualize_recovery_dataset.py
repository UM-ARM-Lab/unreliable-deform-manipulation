#!/usr/bin/env python
import argparse
import pathlib
from time import sleep

import colorama
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from progressbar import progressbar

import rospy
from link_bot_data.recovery_dataset import RecoveryDataset
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import log_scale_0_to_1
from link_bot_pycommon.rviz_animation_controller import RvizSimpleStepper
from moonshine.gpu_config import limit_gpu_mem

limit_gpu_mem(1)


def main():
    colorama.init(autoreset=True)

    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=5)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('--type', choices=['best_to_worst', 'in_order', 'stats'], default='best_to_worst')
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='train')

    args = parser.parse_args()

    rospy.init_node('vis_recovery_dataset')

    dataset = RecoveryDataset(args.dataset_dirs)
    if args.type == 'best_to_worst':
        visualize_best_to_worst(args, dataset)
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


def visualize_best_to_worst(args, dataset: RecoveryDataset):
    tf_dataset = dataset.get_datasets(mode=args.mode)

    # sort the dataset
    examples_to_sort = []
    for example in progressbar(tf_dataset):
        recovery_probability_1 = example['recovery_probability'][1]
        # if recovery_probability_1 > 0.0:
        examples_to_sort.append(example)

    examples_to_sort = sorted(examples_to_sort, key=lambda e: e['recovery_probability'][1], reverse=True)

    scenario = get_scenario(dataset.hparams['scenario'])

    # print("BEST")
    rstepper = RvizSimpleStepper()
    for i, example in enumerate(examples_to_sort):
        visualize_example(scenario, example, dataset.state_keys, dataset.action_keys)
        print(i)
        rstepper.step()

    # print("WORST")
    # for example in examples_to_sort[:10]:
    #     visualize_example(dataset, example)


def visualize_in_order(args, dataset: RecoveryDataset):
    scenario = get_scenario(dataset.hparams['scenario'])
    tf_dataset = dataset.get_datasets(mode=args.mode)

    for example in tf_dataset:
        visualize_example(dataset, example)


def visualize_example(scenario, example, state_keys, action_keys):
    scenario.plot_environment_rviz(example)

    recovery_probability = example['recovery_probability'][1]
    color_factor = log_scale_0_to_1(recovery_probability, k=10)
    scenario.plot_recovery_probability(recovery_probability)
    s_0 = {k: example[k][0] for k in state_keys}
    s_1 = {k: example[k][1] for k in state_keys}
    a = {k: example[k][0] for k in action_keys}

    for i in range(5):
        scenario.plot_action_rviz(s_0, a, label='observed')
        scenario.plot_state_rviz(s_0, label='observed', idx=1, color='w')
        scenario.plot_state_rviz(s_1, label='observed', idx=2, color=cm.Reds(color_factor))
        sleep(0.01)


if __name__ == '__main__':
    main()
