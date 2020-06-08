#!/usr/bin/env python
import argparse
import json
import pathlib
import time
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_classifiers import classifier_utils
from link_bot_classifiers.visualization import visualize_classifier_example, classifier_example_title
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import add_planned
from link_bot_data.recovery_dataset import RecoveryDataset
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.pycommon import print_dict
from moonshine.gpu_config import limit_gpu_mem
from moonshine.moonshine_utils import remove_batch, dict_of_sequences_to_sequence_of_dicts, numpify

limit_gpu_mem(1)


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_type', choices=['classifier', 'recovery'])
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
    args.batch_size = 1

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if args.dataset_type == 'classifier':
        classifier_dataset = ClassifierDataset(args.dataset_dirs, load_true_states=True)
    elif args.dataset_type == 'recovery':
        classifier_dataset = RecoveryDataset(args.dataset_dirs, load_true_states=True)
    else:
        raise NotImplementedError()

    visualize_dataset(args, classifier_dataset)


def visualize_dataset(args, classifier_dataset):
    dataset = classifier_dataset.get_datasets(mode=args.mode, take=args.take)
    scenario = get_scenario(classifier_dataset.hparams['scenario'])
    model_hparams = json.load(args.model_hparams.open("r"))
    classifier_model_dir = pathlib.Path('log_data/collision')
    collision_checker = classifier_utils.load_generic_model(classifier_model_dir, scenario=scenario)
    if args.shuffle:
        dataset = dataset.shuffle(buffer_size=512)
    print_dict(next(iter(dataset)))
    dataset = dataset.batch(1)
    now = int(time.time())
    outdir = pathlib.Path('results') / f'anim_{now}'
    outdir.mkdir(parents=True)
    done = False
    reconverging_count = 0
    positive_count = 0
    negative_count = 0
    count = 0
    iterator = iter(dataset)
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

        example = remove_batch(example)

        is_close = example['is_close'].numpy().squeeze()
        n_close = np.count_nonzero(is_close)
        n_far = is_close.shape[0] - n_close
        positive_count += n_close
        negative_count += n_far
        reconverging = n_far > 0 and is_close[-1]
        environment = numpify(scenario.get_environment_from_example(example))
        predictions = {}
        for state_key in classifier_dataset.state_keys:
            predictions[state_key] = example[add_planned(state_key)]
        predictions = dict_of_sequences_to_sequence_of_dicts(predictions)
        # in_collision = collision_checker.check_constraint(environment=environment,
        #                                                   states_sequence=predictions,
        #                                                   actions=example['action'])[0] < 0.5

        if args.only_reconverging and not reconverging:
            continue

        # if args.only_in_collision and not in_collision:
        #     continue

        if count == 0:
            print_dict(example)

        if reconverging:
            reconverging_count += 1

        count += 1

        title = classifier_example_title(example)

        # Print statistics intermittently
        if count % 100 == 0:
            print_stats_and_timing(args, count, reconverging_count, negative_count, positive_count)

        #############################
        # Show Visualization
        #############################
        # print(example['traj_idx'].numpy()[0],
        #       example['prediction_start_t'].numpy(),
        #       example['classifier_start_t'].numpy(),
        #       example['classifier_end_t'].numpy())
        # if example['prediction_start_t'].numpy() < 1.0:
        #     continue

        valid_seq_length = (example['classifier_end_t'] - example['classifier_start_t'] + 1).numpy()
        if args.at_least_length and valid_seq_length < args.at_least_length:
            continue

        _ = visualize_classifier_example(args=args,
                                         scenario=scenario,
                                         outdir=outdir,
                                         model_hparams=model_hparams,
                                         classifier_dataset=classifier_dataset,
                                         example=example,
                                         example_idx=count,
                                         title=title,
                                         fps=args.fps)
        if not args.no_plot:
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
