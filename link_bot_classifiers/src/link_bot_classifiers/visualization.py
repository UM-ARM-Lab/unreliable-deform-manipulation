import pathlib
import struct
from typing import Optional, Dict, List

import numpy as np
import tensorflow as tf
from colorama import Fore
from matplotlib import cm
from matplotlib import pyplot as plt

import rospy
from link_bot_data.classifier_dataset import ClassifierDataset
from link_bot_data.link_bot_dataset_utils import add_predicted
from link_bot_data.visualization import plot_rope_configuration, plot_arrow
from link_bot_pycommon import link_bot_sdf_utils
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.link_bot_sdf_utils import idx_to_point_3d_in_env
from link_bot_pycommon.rviz_animation_controller import RvizAnimationController
from moonshine.moonshine_utils import numpify, dict_of_sequences_to_sequence_of_dicts, add_batch, remove_batch
from mps_shape_completion_msgs.msg import OccupancyStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header


def visualize_classifier_example_3d(scenario: ExperimentScenario,
                                    example: Dict,
                                    n_time_steps: int):
    time_steps = np.arange(n_time_steps)
    scenario.plot_environment_rviz(example)
    anim = RvizAnimationController(time_steps)
    while not anim.done:
        t = anim.t()
        actual_t = remove_batch(scenario.index_state_time(add_batch(example), t))
        pred_t = remove_batch(scenario.index_predicted_state_time(add_batch(example), t))
        action_t = remove_batch(scenario.index_action_time(add_batch(example), t))
        label_t = remove_batch(scenario.index_label_time(add_batch(example), t)).numpy()
        scenario.plot_state_rviz(actual_t, label='actual', color='#ff0000aa')
        scenario.plot_state_rviz(pred_t, label='predicted', color='#0000ffaa')
        state_action_t = {}
        state_action_t.update(actual_t)
        state_action_t.update(action_t)
        scenario.plot_action_rviz(state_action_t)
        scenario.plot_is_close(label_t)

        # this will return when either the animation is "playing" or because the user stepped forward
        anim.step()


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
                                 accept_probabilities: Optional = None,
                                 end_idx=None,
                                 fps: Optional[int] = 1):
    if args.display_type == 'just_count':
        pass
    elif args.display_type == 'image':
        return trajectory_image_from_example(example, model_hparams, title, accept_probabilities, end_idx)
    elif args.display_type == 'anim':
        anim = trajectory_animation(scenario, classifier_dataset, example, example_idx, accept_probabilities, fps=fps)
        if args.save:
            filename = outdir / f'example_{example_idx}.gif'
            print(Fore.CYAN + f"Saving {filename}" + Fore.RESET)
            anim.save(filename, writer='imagemagick', dpi=100, fps=fps)
        return anim
    elif args.display_type == '2d':
        fig = plt.figure()
        ax = plt.gca()
        assert example["is_close"].shape[0] == 2
        trajectory_plot_from_dataset(ax, classifier_dataset, example, scenario, title)
        return fig


def transition_plot(example, label, title):
    full_env = example['env'].numpy()
    full_env_extent = example['extent'].numpy()
    res = example['res'].numpy()
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
    actual_states = {}
    planned_states = {}
    for state_key in classifier_dataset.state_keys:
        actual_states[state_key] = numpify(example[state_key])
        planned_states[state_key] = numpify(example[add_predicted(state_key)])
    environment = numpify(scenario.get_environment_from_example(example))
    actual_states = dict_of_sequences_to_sequence_of_dicts(actual_states)
    planned_states = dict_of_sequences_to_sequence_of_dicts(planned_states)

    trajectory_plot(ax, scenario, environment, actual_states, planned_states)

    traj_idx = int(example["traj_idx"][0].numpy())
    label = example["is_close"][1]
    title = f"Traj={traj_idx}, label={label}"
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
    T = len(actual_states)
    for time_idx in range(T):
        # don't plot NULL states
        actual_color = cm.Reds_r(time_idx / T)
        planned_color = cm.Blues_r(time_idx / T)
        if actual_states is not None:
            actual_s_t = actual_states[time_idx]
            scenario.plot_state(ax, actual_s_t, color=actual_color, s=20, zorder=2, label='actual state', alpha=0.5)
        if predicted_states is not None:
            planned_s_t = predicted_states[time_idx]
            scenario.plot_state(ax, planned_s_t, color=planned_color, s=5, zorder=3, label='planned state', alpha=0.5)


def trajectory_image_from_example(example, model_hparams, title, accept_probabilities=None, end_idx=None):
    image_key = model_hparams['image_key']

    image = example[image_key].numpy()
    actions = example['action']
    is_close = example['is_close']
    T = image.shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=T)
    trajectory_image(axes=axes,
                     image=image,
                     actions=actions,
                     labels=is_close,
                     accept_probabilities=accept_probabilities,
                     end_idx=end_idx)
    fig.suptitle(title)


def trajectory_image(axes, image, actions=None, labels=None, accept_probabilities=None, end_idx=None):
    T = image.shape[0]
    if end_idx is not None:
        T = min(T, end_idx + 1)
    for t in range(T):
        env_image_t = np.tile(image[t, :, :, -1:], [1, 1, 3])
        state_image_t = state_image_to_cmap(image[t, :, :, :-1])
        image_t = paste_over(state_image_t, env_image_t)
        axes[t].imshow(np.flipud(image_t), vmin=0, vmax=1)
        axes[t].set_title(f"t={t}")
        if t < T - 1:
            if actions is not None:
                axes[t].text(x=10, y=image.shape[1] + 10, s=f"a={actions[t]}")
        if t > 0:
            if accept_probabilities is not None:
                axes[t].text(x=10, y=image.shape[1] + 25, s=f"p={accept_probabilities[t - 1]:0.3f}")
            if labels is not None:
                axes[t].text(x=10, y=image.shape[1] + 40, s=f"label={labels[t]:0.3f}")
    for t in range(image.shape[0]):
        axes[t].set_xticks([])
        axes[t].set_yticks([])


def voxel_grid_to_colored_point_cloud(voxel_grids, environment, frame: str, cmap=cm.viridis, binary_threshold=0.5):
    h, w, c, n = voxel_grids.shape
    points = []
    for idx in range(n):
        voxel_grid = voxel_grids[:, :, :, idx]
        r, g, b = cmap(idx / n)[:3]
        r = int(255 * r)
        g = int(255 * g)
        b = int(255 * b)
        a = 255
        # row, col, channel
        indices = tf.where(voxel_grid > binary_threshold).numpy()
        for row, col, channel in indices:
            x, y, z = idx_to_point_3d_in_env(row, col, channel, environment)
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            pt = [x, y, z, rgb]
            points.append(pt)

    header = Header()
    header.frame_id = frame
    header.stamp = rospy.Time.now()
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('rgb', 12, PointField.UINT32, 1)]

    msg = point_cloud2.create_cloud(header, fields, points)

    return msg


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


def trajectory_animation(scenario, classifier_dataset, example, count, accept_probabilities, fps: Optional[int] = 1):
    # animate the state versus ground truth
    anim = scenario.animate_predictions_from_classifier_dataset(state_keys=classifier_dataset.state_keys,
                                                                example_idx=count,
                                                                dataset_element=example,
                                                                accept_probabilities=accept_probabilities,
                                                                fps=fps)
    return anim


def classifier_example_title(example):
    traj_idx = int(example["traj_idx"][0].numpy())
    start = int(example["classifier_start_t"].numpy())
    end = int(example["classifier_end_t"].numpy())
    title = f"Traj={traj_idx},{start}-{end}"
    return title
