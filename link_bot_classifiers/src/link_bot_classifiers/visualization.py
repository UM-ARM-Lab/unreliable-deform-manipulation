import pathlib
from typing import Optional, Dict, List

import numpy as np
from colorama import Fore
from matplotlib import cm
from matplotlib import pyplot as plt

from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import add_planned, state_dict_is_null
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


def visualize_classifier_example(args,
                                 scenario: ExperimentScenario,
                                 outdir: pathlib.Path,
                                 model_hparams: Dict,
                                 classifier_dataset: ClassifierDataset,
                                 example: Dict,
                                 example_idx: int,
                                 title: str,
                                 accept_probability: Optional[float] = None,
                                 fps: Optional[int] = 1):
    label = example['label'].numpy().squeeze()
    image_key = model_hparams['image_key']
    if args.display_type == 'just_count':
        pass
    elif args.display_type == 'image':
        return trajectory_image_from_example(example, model_hparams, title)
    elif args.display_type == 'anim':
        anim = trajectory_animation(scenario, classifier_dataset, example, example_idx, accept_probability, fps=fps)
        if args.save:
            filename = outdir / f'example_{example_idx}.gif'
            print(Fore.CYAN + f"Saving {filename}" + Fore.RESET)
            anim.save(filename, writer='imagemagick', dpi=100, fps=fps)
        return anim
    elif args.display_type == 'plot':
        if image_key == 'transition_image':
            return transition_plot(example, label, title)
        elif image_key == 'trajectory_image':
            fig = plt.figure()
            ax = plt.gca()
            trajectory_plot_from_dataset(ax, classifier_dataset, example, scenario, title)
            return fig


def transition_plot(example, label, title):
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


def trajectory_plot_from_dataset(ax, classifier_dataset, example, scenario, title):
    actual_states = example[classifier_dataset.label_state_key].numpy()
    planned_states = example[add_planned(classifier_dataset.label_state_key)].numpy()
    environment = scenario.get_environment_from_example(example)

    trajectory_plot(ax, scenario, environment, actual_states, planned_states)

    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


def trajectory_plot(ax,
                    scenario,
                    environment: Dict,
                    actual_states: Optional[List[Dict]] = None,
                    predicted_states: Optional[List[Dict]] = None):
    scenario.plot_environment(ax, environment)
    T = len(predicted_states)
    for time_idx in range(T):
        # don't plot NULL states
        actual_color = cm.Reds_r(time_idx / T)
        planned_color = cm.Blues_r(time_idx / T)
        if not state_dict_is_null(predicted_states[time_idx]):
            if actual_states is not None:
                actual_s_t = actual_states[time_idx]
                scenario.plot_state(ax, actual_s_t, color=actual_color, s=20, zorder=2, label='actual state', alpha=0.5)
            if predicted_states is not None:
                planned_s_t = predicted_states[time_idx]
                scenario.plot_state(ax, planned_s_t, color=planned_color, s=5, zorder=3, label='planned state', alpha=0.5)


def trajectory_image_from_example(example, model_hparams, title):
    image_key = model_hparams['image_key']

    image = example[image_key].numpy()
    actions = example['action']
    T = image.shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=T)
    trajectory_image(axes=axes, image=image, actions=actions)
    fig.suptitle(title)


def trajectory_image(axes, image, actions):
    T = image.shape[0]
    for t in range(T):
        env_image_t = np.tile(image[t, :, :, -1:], [1, 1, 3])
        state_image_t = state_image_to_cmap(image[t, :, :, :-1])
        image_t = paste_over(state_image_t, env_image_t)
        axes[t].imshow(np.flipud(image_t), vmin=0, vmax=1)
        axes[t].set_title(f"t={t}")
        if t < T - 1:
            axes[t].text(x=10, y=image.shape[1] + 10, s=f"u={actions[t]}")
        axes[t].set_xticks([])
        axes[t].set_yticks([])


def state_image_to_cmap(state_image: np.ndarray, cmap=cm.viridis, binary_threshold=0.1):
    h, w, n_channels = state_image.shape
    new_image = np.zeros([h, w, 3])
    for channel_idx in range(n_channels):
        channel = np.take(state_image, indices=channel_idx, axis=-1)
        color = cmap(channel_idx / n_channels)[:3]
        rows, cols = np.where(channel > binary_threshold)
        new_image[rows, cols] = color
    return new_image


def paste_over(i1, i2, binary_threshold=0.1):
    # first create a mask for everywhere i1 > binary_threshold, and zero out those pixels in i2, then add.
    mask = np.any(i1 > binary_threshold, axis=2)
    i2[mask] = 0
    return i2 + i1


def trajectory_animation(scenario, classifier_dataset, example, count, accept_probability, fps: Optional[int] = 1):
    # animate the state versus ground truth
    anim = scenario.animate_predictions_from_classifier_dataset(classifier_dataset=classifier_dataset,
                                                                example_idx=count,
                                                                dataset_element=example,
                                                                accept_probability=accept_probability,
                                                                fps=fps)
    return anim


def classifier_example_title(example):
    label = example['label'].numpy().squeeze()
    traj_idx = int(example["traj_idx"][0].numpy())
    start = int(example["classifier_start_t"].numpy())
    end = int(example["classifier_end_t"].numpy())
    title = f"Traj={traj_idx},{start}-{end} Label={label}"
    return title
