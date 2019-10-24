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
        actual_env,
        actual_env_extent,
        res,
        state,
        next_state,
        title,
        label=None):
    plt.figure()
    ax = plt.gca()

    plt.imshow(np.flipud(planned_env), extent=planned_env_extent, zorder=1, vmin=0, vmax=1, cmap='viridis', alpha=0.5)
    if actual_env is not None:
        plt.imshow(np.flipud(actual_env), extent=actual_env_extent, zorder=1, vmin=0, vmax=1, cmap='viridis', alpha=0.5)
    if state is not None:
        plot_rope_configuration(ax, state, c='red', label='state', zorder=2)
    if next_state is not None:
        plot_rope_configuration(ax, next_state, c='orange', label='next state', zorder=3)

    origin_x, origin_y = link_bot_sdf_utils.idx_to_point(0, 0, res, planned_env_origin)
    plt.scatter(origin_x, origin_y, label='origin', marker='*')

    plot_rope_configuration(ax, planned_state, c='blue', label='planned state', zorder=4)
    plot_rope_configuration(ax, planned_next_state, c='cyan', label='planned next state', zorder=5)
    if state is not None:
        ax.scatter(state[4], state[5], c='k')
    ax.scatter(planned_state[4], planned_state[5], c='k')

    if label is not None:
        label_color = 'g' if label else 'r'
        plt.plot([-2.5, 2.5, 2.5, -2.5, -2.5], [-2.5, -2.5, 2.5, 2.5, -2.5], c=label_color, linewidth=6)

    plt.axis("equal")
    plt.xlim(-2.5, 2.5)
    # plt.ylim(-2.5, 2.5)
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
