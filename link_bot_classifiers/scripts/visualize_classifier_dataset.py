#!/usr/bin/env python
import argparse
import pathlib
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import NULL_PAD_VALUE, add_all, add_all_and_planned, add_planned, add_next_and_planned
from link_bot_planning.get_scenario import get_scenario
from moonshine.image_functions import partial_add_traj_image
from moonshine.numpy_utils import remove_batch

tf.compat.v1.enable_eager_execution()


def main():
    plt.style.use("./classifier.mplstyle")
    np.set_printoptions(suppress=True, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('display_type',
                        choices=['just_count', 'transition_image', 'transition_plot', 'trajectory_image', 'trajectory_plot'])
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pre', type=int, default=0.15)
    parser.add_argument('--post', type=int, default=0.21)
    parser.add_argument('--discard-pre-far', action='store_true')
    parser.add_argument('--action-in-image', action='store_true')
    parser.add_argument('--take', type=int)
    parser.add_argument('--local-env-s', type=int, default=100)
    parser.add_argument('--rope-image-k', type=float, default=1000.0)
    parser.add_argument('--only-negative', action='store_true')
    parser.add_argument('--perf', action='store_true', help='print time per iteration')
    parser.add_argument('--no-plot', action='store_true', help='only print statistics')

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.compat.v1.random.set_random_seed(args.seed)

    states_keys = ['link_bot']

    classifier_dataset = ClassifierDataset(args.dataset_dirs)
    dataset = classifier_dataset.get_datasets(mode=args.mode, take=args.take)
    scenario = get_scenario(classifier_dataset.hparams['scenario'])

    # if args.display_type == 'transition_image':
    #     dataset = add_transition_image(dataset,
    #                                    states_keys=states_keys,
    #                                    action_in_image=args.action_in_image,
    #                                    scenario=scenario,
    #                                    local_env_h=args.local_env_s,
    #                                    local_env_w=args.local_env_s,
    #                                    rope_image_k=args.rope_image_k)
    postprocess = partial_add_traj_image(states_keys=states_keys, batch_size=1, rope_image_k=args.rope_image_k)

    if args.shuffle:
        dataset = dataset.shuffle(buffer_size=1024)

    dataset = dataset.batch(1)

    done = False

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

        example = postprocess(example)
        example = remove_batch(example)

        label = example['label'].numpy().squeeze()

        if args.only_negative and label != 0:
            continue
        if label:
            positive_count += 1
        else:
            negative_count += 1

        count += 1

        if args.no_plot:
            # still count, but do nothing else
            continue

        title = make_title(example, label)

        #############################
        # Show Visualization
        #############################
        show_visualization(args, classifier_dataset, example, label, scenario, title)

    total_dt = perf_counter() - t0

    print_stats_and_timing(args, count, negative_count, positive_count, total_dt)


def print_stats_and_timing(args, count, negative_count, positive_count, total_dt):
    if args.perf:
        print("Total iteration time = {:.4f}".format(total_dt))
    class_balance = positive_count / count * 100
    print("Number of examples: {}".format(count))
    print("Number positive: {}".format(positive_count))
    print("Number negative: {}".format(negative_count))
    print("Class balance: {:4.1f}% positive".format(class_balance))


def show_visualization(args, classifier_dataset, example, label, scenario, title):
    if args.display_type == 'just_count':
        pass
    elif args.display_type == 'transition_image':
        show_transition_image(example, scenario, title)
    elif args.display_type == 'trajectory_image':
        show_trajectory_image(example, title)
    elif args.display_type == 'trajectory_plot':
        show_trajectory_plot(classifier_dataset, example, scenario, title)
    elif args.display_type == 'transition_plot':
        show_transition_plot(example, label, title)


def show_transition_plot(example, label, title):
    full_env = example['full_env/env'].numpy()
    full_env_extent = example['full_env/extent'].numpy()
    res = example['full_env/res'].numpy()
    state = example['link_bot'].numpy()
    action = example['action'].numpy()
    next_state = example['link_bot_next'].numpy()
    planned_next_state = example['planned_state/link_bot_next'].numpy()
    plot_classifier_data(
        next_state=next_state,
        action=action,
        planned_next_state=planned_next_state,
        res=res,
        state=state,
        actual_env=full_env,
        actual_env_extent=full_env_extent,
        title=title,
        label=label)
    plt.legend()
    plt.show(block=True)


def show_trajectory_plot(classifier_dataset, example, scenario, title):
    full_env = example['full_env/env'].numpy()
    full_env_extent = example['full_env/extent'].numpy()
    actual_state_all = example[add_all(classifier_dataset.label_state_key)].numpy()
    planned_state_all = example[add_all_and_planned(classifier_dataset.label_state_key)].numpy()
    plt.figure()
    plt.imshow(np.flipud(full_env), extent=full_env_extent)
    ax = plt.gca()
    for time_idx in range(planned_state_all.shape[0]):
        # don't plot NULL states
        if not np.any(planned_state_all[time_idx, 0] == NULL_PAD_VALUE):
            actual_state = {
                classifier_dataset.label_state_key: actual_state_all[time_idx]
            }
            planned_state = {
                classifier_dataset.label_state_key: planned_state_all[time_idx]
            }
            scenario.plot_state(ax, actual_state, color='red', s=20, zorder=2, label='actual state')
            scenario.plot_state(ax, planned_state, color='blue', s=5, zorder=3, label='planned state')
    if scenario == 'tether':
        start_t = int(example['start_t'].numpy())
        tether_start = example[add_all('tether')][start_t].numpy()
        tether_start_state = {
            'tether': tether_start
        }
        scenario.plot_state(ax, tether_start_state, color='green', s=5, zorder=1)
    plt.title(title)
    plt.show()


def show_trajectory_image(example, title):
    image = example['trajectory_image'].numpy()
    n_channels = image.shape[2]
    plt.figure()
    plt.title(title)
    for c in range(n_channels):
        plt.subplot()
        plt.imshow(np.flipud(image[:, :, c]))
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show(block=True)


def show_transition_image(example, scenario, title):
    image = example['transition_image'].numpy()
    n_channels = image.shape[2]
    plt.figure()
    plt.title(title)
    for c in range(n_channels):
        plt.subplot()
        plt.imshow(np.flipud(image[:, :, c]))
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show(block=True)


def make_title(example, label):
    if add_planned('stdev') in example:
        stdev = example[add_planned('stdev')].numpy().squeeze()
        stdev_next = example[add_next_and_planned('stdev')].numpy().squeeze()
        title = "Label = {}, stdev={:.3f},{:.3f}".format(label, stdev, stdev_next)
    else:
        title = "Label = {}, no stdev".format(label)
    return title


if __name__ == '__main__':
    main()
