#!/usr/bin/env python
import argparse
import pathlib
from time import perf_counter

import colorama
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import stats

import rospy
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import add_predicted
from link_bot_pycommon.pycommon import print_dict
from link_bot_pycommon.rviz_animation_controller import RvizSimpleStepper
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import remove_batch
from std_msgs.msg import Float32

limit_gpu_mem(1)


def main():
    colorama.init(autoreset=True)

    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('--display-type', choices=['just_count', '3d', 'stdev'], default='3d')
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='train')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--take', type=int)
    parser.add_argument('--only-negative', action='store_true')
    parser.add_argument('--only-positive', action='store_true')
    parser.add_argument('--only-in-collision', action='store_true')
    parser.add_argument('--only-reconverging', action='store_true')
    parser.add_argument('--perf', action='store_true', help='print time per iteration')
    parser.add_argument('--no-plot', action='store_true', help='only print statistics')

    args = parser.parse_args()
    args.batch_size = 1

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    rospy.init_node("visualize_classifier_data")

    classifier_dataset = ClassifierDataset(args.dataset_dirs, load_true_states=True)

    visualize_dataset(args, classifier_dataset)


def visualize_dataset(args, classifier_dataset):
    tf_dataset = classifier_dataset.get_datasets(mode=args.mode, take=args.take)
    scenario = classifier_dataset.scenario

    tf_dataset = tf_dataset.batch(1)

    iterator = iter(tf_dataset)
    t0 = perf_counter()

    stdev_pub_ = rospy.Publisher("stdev", Float32, queue_size=10)

    reconverging_count = 0
    positive_count = 0
    negative_count = 0
    count = 0

    stdevs = []
    labels = []
    stdevs_for_negative = []
    stdevs_for_positive = []

    done = False
    stepper = RvizSimpleStepper()
    while not done:
        iter_t0 = perf_counter()
        try:
            example = next(iterator)
        except StopIteration:
            break
        iter_dt = perf_counter() - iter_t0
        if args.perf:
            print("{:6.4f}".format(iter_dt))

        example = remove_batch(example)

        is_close = example['is_close'].numpy().squeeze()
        count += is_close.shape[0]

        n_close = np.count_nonzero(is_close[-1])
        n_far = is_close.shape[0] - n_close
        positive_count += n_close
        negative_count += n_far
        reconverging = n_far > 0 and is_close[-1]

        if args.only_reconverging and not reconverging:
            continue

        if args.only_negative and np.any(is_close[1:]):
            continue

        if count == 0:
            print_dict(example)

        if reconverging:
            reconverging_count += 1

        # Print statistics intermittently
        if count % 1000 == 0:
            print_stats_and_timing(args, count, reconverging_count, negative_count, positive_count)

        #############################
        # Show Visualization
        #############################
        if args.display_type == 'just_count':
            continue
        elif args.display_type == '3d':
            # print(example['is_close'])
            if example['is_close'][0] == 0:
                continue
            scenario.plot_transition_rviz(classifier_dataset.hparams, example, 0)
            stdev_t = example[add_predicted('stdev')][0, 0].numpy()
            stdev_msg = Float32()
            stdev_msg.data = stdev_t
            stdev_pub_.publish(stdev_msg)

            stepper.step()

        elif args.display_type == 'stdev':
            for t in range(1, classifier_dataset.horizon):
                stdev_t = example[add_predicted('stdev')][t, 0].numpy()
                label_t = example['is_close'][t]
                stdevs.append(stdev_t)
                labels.append(label_t)
                if label_t > 0.5:
                    stdevs_for_positive.append(stdev_t)
                else:
                    stdevs_for_negative.append(stdev_t)
        else:
            raise NotImplementedError()
    total_dt = perf_counter() - t0

    if args.display_type == 'stdev':
        print(f"p={stats.f_oneway(stdevs_for_negative, stdevs_for_positive)[1]}")

        plt.figure()
        plt.title(" ".join([str(d.name) for d in args.dataset_dirs]))
        bins = plt.hist(stdevs_for_negative, label='negative examples', alpha=0.8, density=True)[1]
        plt.hist(stdevs_for_positive, label='positive examples', alpha=0.8, bins=bins, density=True)
        plt.ylabel("count")
        plt.xlabel("stdev")
        plt.legend()
        plt.show()

    print_stats_and_timing(args, count, reconverging_count, negative_count, positive_count, total_dt)


def print_stats_and_timing(args, count, reconverging_count, negative_count, positive_count, total_dt=None):
    if args.perf and total_dt is not None:
        print("Total iteration time = {:.4f}".format(total_dt))
    class_balance = positive_count / count * 100
    print("Number of examples: {}".format(count))
    print("Number of reconverging examples: {}".format(reconverging_count))
    print("Number positive: {}".format(positive_count))
    print("Number negative: {}".format(negative_count))
    print("Class balance: {:4.1f}% positive".format(class_balance))


if __name__ == '__main__':
    main()
