#!/usr/bin/env python
import argparse
import json
import pathlib
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import NULL_PAD_VALUE, add_all, add_planned, state_dict_is_null_tf
from link_bot_planning.get_scenario import get_scenario
from moonshine.image_functions import setup_image_inputs
from moonshine.moonshine_utils import remove_batch


def main():
    plt.style.use("slides")
    np.set_printoptions(suppress=True, linewidth=200, precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('model_hparams', type=pathlib.Path, help='classifier model hparams')
    parser.add_argument('display_type', choices=['just_count', 'image', 'plot'])
    parser.add_argument('--mode', choices=['train', 'val', 'test', 'all'], default='train')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--no-balance', action='store_true')
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
        if label:
            positive_count += 1
        else:
            negative_count += 1

        if args.only_funneling and not funneling:
            continue

        count += 1

        if args.no_plot:
            # still count, but do nothing else
            continue

        title = make_title(example, label)

        #############################
        # Show Visualization
        #############################
        show_visualization(args, model_hparams, classifier_dataset, example, label, scenario, title)

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


def show_visualization(args, model_hparams, classifier_dataset, example, label, scenario, title):
    image_key = model_hparams['image_key']
    if args.display_type == 'just_count':
        pass
    elif args.display_type == 'image':
        show_image(args, example, model_hparams, title)
    elif args.display_type == 'plot':
        if image_key == 'transition_image':
            show_transition_plot(example, label, title)
        elif image_key == 'trajectory_image':
            show_trajectory_plot(classifier_dataset, example, scenario, title)


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
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show(block=True)


def show_trajectory_plot(classifier_dataset, example, scenario, title):
    full_env = example['full_env/env'].numpy()
    full_env_extent = example['full_env/extent'].numpy()
    actual_state_all = example[classifier_dataset.label_state_key].numpy()
    planned_state_all = example[add_planned(classifier_dataset.label_state_key)].numpy()
    plt.figure()
    plt.imshow(np.flipud(full_env), extent=full_env_extent)
    ax = plt.gca()
    for time_idx in range(planned_state_all.shape[0]):
        # don't plot NULL states
        if not np.any(planned_state_all[time_idx] == NULL_PAD_VALUE):
            actual_state = {
                classifier_dataset.label_state_key: actual_state_all[time_idx]
            }
            planned_state = {
                classifier_dataset.label_state_key: planned_state_all[time_idx]
            }
            scenario.plot_state(ax, actual_state, color='red', s=20, zorder=2, label='actual state', alpha=0.2)
            scenario.plot_state(ax, planned_state, color='blue', s=5, zorder=3, label='planned state', alpha=0.2)
    if scenario == 'tether':
        start_t = int(example['start_t'].numpy())
        tether_start = example[add_all('tether')][start_t].numpy()
        tether_start_state = {
            'tether': tether_start
        }
        scenario.plot_state(ax, tether_start_state, color='green', s=5, zorder=1)
    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


def show_image(args, example, model_hparams, title):
    image_key = model_hparams['image_key']
    valid_seq_length = (example['classifier_end_t'] - example['classifier_start_t'] + 1).numpy()

    if args.only_length and args.only_length != valid_seq_length:
        return

    image = example[image_key].numpy()
    n_channels = image.shape[2]
    if n_channels != 3:
        T = image.shape[0]
        fig, axes = plt.subplots(nrows=1, ncols=T)
        fig.suptitle(title)
        for t in range(T):
            env_image_t = image[t, :, :, -1]
            state_image_t = np.sum(image[t, :, :, :-1], axis=2)
            zeros = np.zeros_like(env_image_t)
            image_t = np.stack([env_image_t, state_image_t, zeros], axis=2)
            axes[t].imshow(np.flipud(image_t), vmin=0, vmax=1)
            axes[t].set_title(f"t={t}")
            axes[t].set_xticks([])
            axes[t].set_yticks([])
    else:
        plt.imshow(np.flipud(image))
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(title)

    plt.show(block=True)


def make_title(example, label):
    traj_idx = int(example["traj_idx"][0].numpy())
    start = int(example["classifier_start_t"].numpy())
    end = int(example["classifier_end_t"].numpy())
    title = f"Traj={traj_idx},{start}-{end} Label={label}"
    return title


if __name__ == '__main__':
    main()
