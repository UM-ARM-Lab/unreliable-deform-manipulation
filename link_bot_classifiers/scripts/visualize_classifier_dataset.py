#!/usr/bin/env python
import argparse
import json
import pathlib
import time
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_classifiers.visualization import visualize_classifier_example, classifier_example_title
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_pycommon.link_bot_pycommon import print_dict
from moonshine.gpu_config import limit_gpu_mem
from moonshine.image_functions import setup_image_inputs
from moonshine.moonshine_utils import remove_batch

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
    parser.add_argument('--no-balance', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--only-length', type=int)
    parser.add_argument('--take', type=int)
    parser.add_argument('--only-negative', action='store_true')
    parser.add_argument('--only-funneling', action='store_true')
    parser.add_argument('--perf', action='store_true', help='print time per iteration')
    parser.add_argument('--no-plot', action='store_true', help='only print statistics')

    args = parser.parse_args()
    args.batch_size = 1

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    classifier_dataset = ClassifierDataset(args.dataset_dirs, no_balance=args.no_balance)
    dataset = classifier_dataset.get_datasets(mode=args.mode, take=args.take)

    scenario = get_scenario(classifier_dataset.hparams['scenario'])
    model_hparams = json.load(args.model_hparams.open("r"))

    postprocess, _ = setup_image_inputs(args, scenario, classifier_dataset, model_hparams)

    if args.shuffle:
        dataset = dataset.shuffle(buffer_size=512)

    dataset = dataset.batch(1)

    now = int(time.time())
    outdir = pathlib.Path('results') / f'anim_{now}'
    outdir.mkdir(parents=True)

    done = False

    funneling_count = 0
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

        if postprocess is not None:
            example = postprocess(example)

        example = remove_batch(example)

        is_close = example['is_close'].numpy().squeeze()
        last_valid_idx = int(example['last_valid_idx'].numpy().squeeze())
        n_valid_states = last_valid_idx + 1
        valid_is_close = is_close[:last_valid_idx + 1]
        num_diverged = n_valid_states - np.count_nonzero(valid_is_close)
        funneling = num_diverged > 0 and valid_is_close[-1]
        label = example['label'].numpy().squeeze()

        if args.only_negative and label != 0:
            continue
        if args.only_funneling and not funneling:
            continue

        if count == 0:
            print_dict(example)

        if label:
            positive_count += 1
        else:
            negative_count += 1
        if funneling:
            funneling_count += 1

        count += 1

        title = classifier_example_title(example)

        # Print statistics intermittently
        if count % 100 == 0:
            print_stats_and_timing(args, count, funneling_count, negative_count, positive_count)

        #############################
        # Show Visualization
        #############################
        # print(example['traj_idx'].numpy()[0],
        #       example['prediction_start_t'].numpy(),
        #       example['classifier_start_t'].numpy(),
        #       example['classifier_end_t'].numpy())
        valid_seq_length = (example['classifier_end_t'] - example['classifier_start_t'] + 1).numpy()
        if args.only_length and args.only_length != valid_seq_length:
            continue

        _ = visualize_classifier_example(args, scenario, outdir, model_hparams, classifier_dataset, example, count, title)
        if not args.no_plot:
            plt.show()
        else:
            plt.close()

    total_dt = perf_counter() - t0

    print_stats_and_timing(args, count, funneling_count, negative_count, positive_count, total_dt)


def print_stats_and_timing(args, count, funneling_count, negative_count, positive_count, total_dt=None):
    if args.perf and total_dt is not None:
        print("Total iteration time = {:.4f}".format(total_dt))
    class_balance = positive_count / count * 100
    print("Number of examples: {}".format(count))
    print("Number of funneling examples: {}".format(funneling_count))
    print("Number positive: {}".format(positive_count))
    print("Number negative: {}".format(negative_count))
    print("Class balance: {:4.1f}% positive".format(class_balance))


if __name__ == '__main__':
    main()
