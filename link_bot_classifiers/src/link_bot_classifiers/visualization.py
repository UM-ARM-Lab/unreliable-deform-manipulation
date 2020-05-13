import pathlib
from typing import Optional, Dict

import numpy as np
from colorama import Fore
from matplotlib import pyplot as plt

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import add_planned, NULL_PAD_VALUE, add_all
from link_bot_data.visualization import plot_rope_configuration, plot_arrow
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.experiment_scenario import ExperimentScenario


def plot_classifier_data(
        planned_env: Optional = None,
        planned_env_extent: Optional = None,
        planned_state: Optional = None,
        planned_next_state: Optional = None,
        planned_env_origin: Optional = None,
        res: Optional = None,
        state: Optional = None,
        next_state: Optional = None,
        title='',
        action: Optional = None,
        actual_env: Optional = None,
        actual_env_extent: Optional = None,
        label: Optional = None,
        ax: Optional = None):
    # TODO: use scenario plotting here
    if ax is None:
        plt.figure()
        ax = plt.gca()

    if actual_env is not None:
        ax.imshow(np.flipud(actual_env), extent=actual_env_extent, zorder=1, cmap='Greys')
    if state is not None:
        plot_rope_configuration(ax, state, c='red', label='state', zorder=2, linewidth=5)
    if next_state is not None:
        plot_rope_configuration(ax, next_state, c='orange', label='next state', zorder=4, linestyle='--', linewidth=5)
    if state is not None and action is not None:
        plot_arrow(ax, state[-2], state[-1], action[0] / 2, action[1] / 2, color='cyan', linewidth=3)

    if planned_env_origin is not None and res is not None:
        origin_x, origin_y = link_bot_sdf_utils.idx_to_point(0, 0, res, planned_env_origin)
        ax.scatter(origin_x, origin_y, label='origin', marker='*')

    if planned_state is not None:
        plot_rope_configuration(ax, planned_state, c='green', label='planned state', zorder=3)
    if planned_next_state is not None:
        plot_rope_configuration(ax, planned_next_state, c='blue', label='planned next state', zorder=5,
                                linewidth=5)
    if state is not None:
        ax.scatter(state[-2], state[-1], c='k')
    if planned_state is not None:
        ax.scatter(planned_state[-2], planned_state[-1], c='k')

    if label is not None and planned_env_extent is not None:
        label_color = 'g' if label else 'r'
        ax.plot(
            [planned_env_extent[0], planned_env_extent[0], planned_env_extent[1], planned_env_extent[1], planned_env_extent[0]],
            [planned_env_extent[2], planned_env_extent[3], planned_env_extent[3], planned_env_extent[2], planned_env_extent[2]],
            c=label_color, linewidth=4)

    ax.axis("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    # ax.legend()


def make_interpretable_image(image: np.ndarray, n_points: int):
    image_23d = image.squeeze()
    assert (image.shape[2] == 23)
    pre_rope = np.sum(image_23d[:, :, 0:n_points], axis=2)
    post_rope = np.sum(image_23d[:, :, n_points:2 * n_points], axis=2)
    local_env = image_23d[:, :, 2 * n_points]
    interpretable_image = np.stack([pre_rope, post_rope, local_env], axis=2)
    return interpretable_image


def visualize_classifier_example(args,
                                 scenario: ExperimentScenario,
                                 outdir: pathlib.Path,
                                 model_hparams: Dict,
                                 classifier_dataset: ClassifierDataset,
                                 example: Dict,
                                 example_idx: int,
                                 title: str,
                                 accept_probability: Optional[float] = None):
    label = example['label'].numpy().squeeze()
    image_key = model_hparams['image_key']
    if args.display_type == 'just_count':
        pass
    elif args.display_type == 'image':
        return show_image(args, example, model_hparams, title)
    elif args.display_type == 'anim':
        anim = show_anim(args, scenario, classifier_dataset, example, example_idx, accept_probability)
        if args.save:
            filename = outdir / f'example_{example_idx}.gif'
            print(Fore.CYAN + f"Saving {filename}" + Fore.RESET)
            anim.save(filename, writer='imagemagick', dpi=100, fps=1)
        return anim
    elif args.display_type == 'plot':
        if image_key == 'transition_image':
            return show_transition_plot(args, example, label, title)
        elif image_key == 'trajectory_image':
            return show_trajectory_plot(args, classifier_dataset, example, scenario, title)


def show_transition_plot(args, example, label, title):
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


def show_trajectory_plot(args, classifier_dataset, example, scenario, title):
    full_env = example['full_env/env'].numpy()
    full_env_extent = example['full_env/extent'].numpy()
    actual_state_all = example[classifier_dataset.label_state_key].numpy()
    planned_state_all = example[add_planned(classifier_dataset.label_state_key)].numpy()
    fig = plt.figure()
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
    return fig


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
        return fig
    else:
        plt.imshow(np.flipud(image))
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(title)


def show_anim(args, scenario, classifier_dataset, example, count, accept_probability):
    # animate the state versus ground truth
    anim = scenario.animate_predictions_from_classifier_dataset(classifier_dataset=classifier_dataset,
                                                                example_idx=count,
                                                                dataset_element=example,
                                                                accept_probability=accept_probability)
    return anim


def classifier_example_title(example):
    label = example['label'].numpy().squeeze()
    traj_idx = int(example["traj_idx"][0].numpy())
    start = int(example["classifier_start_t"].numpy())
    end = int(example["classifier_end_t"].numpy())
    title = f"Traj={traj_idx},{start}-{end} Label={label}"
    return title
