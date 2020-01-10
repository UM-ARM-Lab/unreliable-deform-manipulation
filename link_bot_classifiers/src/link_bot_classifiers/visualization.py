import numpy as np
from matplotlib import pyplot as plt

from link_bot_data.visualization import plot_rope_configuration
from link_bot_pycommon import link_bot_sdf_utils


def plot_classifier_data(
        planned_env,
        planned_env_extent,
        planned_state,
        planned_next_state,
        planned_env_origin,
        res,
        state,
        next_state,
        title='',
        action=None,
        actual_env=None,
        actual_env_extent=None,
        label=None,
        ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    if planned_env is not None:
        ax.imshow(np.flipud(planned_env), extent=planned_env_extent, zorder=1, vmin=0, vmax=1, cmap='viridis', alpha=0.5)
    if actual_env is not None:
        ax.imshow(np.flipud(actual_env), extent=actual_env_extent, zorder=1, vmin=0, vmax=1, cmap='viridis', alpha=0.5)
    if state is not None:
        plot_rope_configuration(ax, state, c='red', label='state', zorder=2, linewidth=3)
    if next_state is not None:
        plot_rope_configuration(ax, next_state, c='orange', label='next state', zorder=4, linestyle='--', linewidth=3)
    if state is not None and action is not None:
        ax.quiver(state[-2], state[-1], action[0], action[1], width=0.001, scale=6)
    if planned_state is not None and action is not None:
        ax.quiver(planned_state[-2], planned_state[-1], action[0], action[1], width=0.001, scale=6)

    if state is not None and next_state is not None:
        ax.plot([state[-2], next_state[-2]], [state[-1], next_state[-1]], c='w', linewidth=1)

    if planned_env_origin is not None and res is not None:
        origin_x, origin_y = link_bot_sdf_utils.idx_to_point(0, 0, res, planned_env_origin)
        ax.scatter(origin_x, origin_y, label='origin', marker='*')

    if planned_state is not None:
        plot_rope_configuration(ax, planned_state, c='blue', label='planned state', zorder=3)
    if planned_next_state is not None:
        plot_rope_configuration(ax, planned_next_state, c='cyan', label='planned next state', zorder=5, linestyle='-.')
    if state is not None:
        ax.scatter(state[-2], state[-1], c='k')
    if planned_state is not None:
        ax.scatter(planned_state[-2], planned_state[-1], c='k')

    if label is not None:
        label_color = 'g' if label else 'r'
        ax.plot([-2.5, 2.5, 2.5, -2.5, -2.5], [-2.5, -2.5, 2.5, 2.5, -2.5], c=label_color, linewidth=6)

    ax.axis("equal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
