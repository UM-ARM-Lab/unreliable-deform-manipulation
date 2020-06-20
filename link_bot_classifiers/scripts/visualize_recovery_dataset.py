#!/usr/bin/env python
import argparse
import pathlib
import time
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_classifiers.visualization import trajectory_plot
from link_bot_data.recovery_dataset import RecoveryDataset
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import print_dict
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import numpify, dict_of_sequences_to_sequence_of_dicts, remove_batch

limit_gpu_mem(1)


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('model_hparams', type=pathlib.Path, help='classifier model hparams')
    parser.add_argument('display_type', choices=['just_count', 'image', 'anim', 'plot'])
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='train')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fps', type=int, default=1)
    parser.add_argument('--at-least-length', type=int)
    parser.add_argument('--take', type=int)
    parser.add_argument('--only-negative', action='store_true')
    parser.add_argument('--only-positive', action='store_true')
    parser.add_argument('--only-in-collision', action='store_true')
    parser.add_argument('--only-reconverging', action='store_true')
    parser.add_argument('--perf', action='store_true', help='print time per iteration')
    parser.add_argument('--no-plot', action='store_true', help='only print statistics')

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    dataset = RecoveryDataset(args.dataset_dirs, load_true_states=True)

    visualize_dataset(args, dataset)


def visualize_dataset(args, dataset):
    tf_dataset = dataset.get_datasets(mode=args.mode, take=args.take)

    scenario = get_scenario(dataset.hparams['scenario'])
    if args.shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=512)
    now = int(time.time())
    outdir = pathlib.Path('results') / f'anim_{now}'
    outdir.mkdir(parents=True)
    done = False
    reconverging_count = 0
    positive_count = 0
    negative_count = 0
    count = 0
    iterator = iter(tf_dataset)
    t0 = perf_counter()
    while not done:
        iter_t0 = perf_counter()
        try:
            example = next(iterator)
        except StopIteration:
            break
        iter_dt = perf_counter() - iter_t0
        if args.perf:
            print("{:6.4f}".format(iter_dt))

        if count == 0:
            print_dict(example)

        count += 1

        # Print statistics intermittently
        if count % 100 == 0:
            print_stats_and_timing(args, count, reconverging_count, negative_count, positive_count)

        #############################
        # Show Visualization
        #############################
        if not args.no_plot:
            plt.figure()
            ax = plt.gca()
            actual_states = {}
            for state_key in dataset.state_keys:
                actual_states[state_key] = remove_batch(numpify(example[state_key]))
            actual_states = dict_of_sequences_to_sequence_of_dicts(actual_states)

            # FIXME: why does environment have a time dimension???
            environment = remove_batch(numpify(scenario.get_environment_from_example(example)))
            environment['env'] = tf.expand_dims(environment['env'], axis=2)
            trajectory_plot(ax, scenario, environment, actual_states)

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

            plt.show()
        else:
            plt.close()
    total_dt = perf_counter() - t0
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
