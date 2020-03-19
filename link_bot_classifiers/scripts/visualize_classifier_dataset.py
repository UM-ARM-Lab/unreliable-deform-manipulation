#!/usr/bin/env python
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from link_bot_classifiers.visualization import plot_classifier_data
from link_bot_data.link_bot_dataset_utils import balance, add_traj_image, add_transition_image
from link_bot_data.classifier_dataset import ClassifierDataset
# from link_bot_data.old_classifier_dataset import ClassifierDataset
from link_bot_data.visualization import plot_rope_configuration
from link_bot_planning.get_scenario import get_scenario
from link_bot_planning.params import LocalEnvParams

tf.compat.v1.enable_eager_execution()


def main():
    plt.style.use("./classifier.mplstyle")
    np.set_printoptions(suppress=True, linewidth=200)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dirs', type=pathlib.Path, nargs='+')
    parser.add_argument('labeling_params', type=pathlib.Path)
    parser.add_argument('display_type',
                        choices=['just_image', 'transition_image', 'transition_plot', 'trajectory_image', 'trajectory_plot'])
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--scenario', choices=['link_bot', 'tether'])
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pre', type=int, default=0.15)
    parser.add_argument('--post', type=int, default=0.21)
    parser.add_argument('--discard-pre-far', action='store_true')
    parser.add_argument('--action-in-image', action='store_true')
    parser.add_argument('--take', type=int)
    parser.add_argument('--local-env-s', type=int, default=50)
    parser.add_argument('--no-balance', action='store_true')
    parser.add_argument('--only-negative', action='store_true')
    parser.add_argument('--no-plot', action='store_true', help='only print statistics')

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.compat.v1.random.set_random_seed(args.seed)

    scenario = get_scenario(args.scenario)

    labeling_params = json.load(args.labeling_params.open("r"))

    states_keys = ['link_bot']

    classifier_dataset = ClassifierDataset(args.dataset_dirs, labeling_params)
    dataset = classifier_dataset.get_datasets(mode=args.mode, take=args.take)

    if not args.no_balance:
        dataset = balance(dataset)

    if args.display_type == 'transition_image':
        dataset = add_transition_image(dataset,
                                       states_keys=states_keys,
                                       action_in_image=args.action_in_image,
                                       scenario=scenario,
                                       local_env_h=args.local_env_s,
                                       local_env_w=args.local_env_s)
    if args.display_type == 'trajectory_image':
        dataset = add_traj_image(dataset)

    if args.shuffle:
        dataset = dataset.shuffle(buffer_size=1024)

    done = False

    positive_count = 0
    negative_count = 0
    count = 0
    for i, example in enumerate(dataset):

        if i < 234:
            continue
        print(i)

        if done:
            break

        label = int(example['label'].numpy().squeeze())

        if args.only_negative and label != 0:
            continue

        if label:
            positive_count += 1
        else:
            negative_count += 1

        count += 1

        if args.no_plot:
            continue

        # FIXME: make this support arbitrary state keys
        if args.display_type == 'just_image':
            image = example['image'].numpy()
            fig, axes = plt.subplots(1, 3)
            axes[0].imshow(np.flipud(image[:, :, 22]), cmap='Greys')
            axes[0].set_title("Local Environment", fontsize=28)
            axes[1].imshow(np.clip(np.flipud(np.sum(image[:, :, 0:11], axis=2)), 0, 1), cmap='Greys')
            axes[1].set_title("$\hat{s}^t$")
            axes[2].imshow(np.clip(np.flipud(np.sum(image[:, :, 11:22], axis=2)), 0, 1), cmap='Greys')
            axes[2].set_title("$\hat{s}^{t+1}$")
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            plt.show(block=True)
        elif args.display_type == 'transition_image':
            image = example['transition_image'].numpy()
            plt.imshow(np.flipud(image))
            title = "Label = {:d}".format(label),
            plt.title(title)
            plt.show(block=True)
        elif args.display_type == 'trajectory_image':
            image = example['trajectory_image'].numpy()
            plt.imshow(np.flipud(image))
            ax = plt.gca()
            title = "Label = {:d}".format(label)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.title(title)
            plt.show(block=True)
        elif args.display_type == 'trajectory_plot':
            full_env = example['full_env/env'].numpy()
            full_env_extent = example['full_env/extent'].numpy()
            link_bot_state_all = example['planned_state/link_bot_all'].numpy()
            stop_idx = example['planned_state/stop_idx'].numpy()
            actual_link_bot_state_all = example['link_bot_all'].numpy()

            plt.figure()
            plt.imshow(np.flipud(full_env), extent=full_env_extent)
            ax = plt.gca()
            for i in range(stop_idx):
                actual_state = actual_link_bot_state_all[i]
                planned_state = link_bot_state_all[i]
                plot_rope_configuration(ax, actual_state, c='white', s=8)
                plot_rope_configuration(ax, planned_state, c='orange', s=6)
            plt.show()
        elif args.display_type == 'transition_plot':
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
                title="Label = {:d}".format(label),
                label=label)
            plt.legend()
            plt.tight_layout()
            plt.show(block=True)

    class_balance = positive_count / count * 100
    print("Number of examples: {}".format(count))
    print("Number positive: {}".format(positive_count))
    print("Number negative: {}".format(negative_count))
    print("Class balance: {:4.1f}% positive".format(class_balance))


if __name__ == '__main__':
    main()
