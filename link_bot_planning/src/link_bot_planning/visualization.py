import numpy as np
from matplotlib import pyplot as plt

from link_bot_data.visualization import plot_rope_configuration


def plot_classifier_data(
        planned_sdf,
        planned_sdf_extent,
        planned_state,
        planned_next_state,
        actual_sdf,
        actual_sdf_extent,
        state,
        next_state,
        title,
        label=None):

    plt.figure()
    ax = plt.gca()

    plt.imshow(np.flipud(planned_sdf) > 0, extent=planned_sdf_extent, zorder=1, vmin=0, vmax=1, cmap='viridis')
    if actual_sdf is not None:
        plt.imshow(np.flipud(actual_sdf) > 0, extent=actual_sdf_extent, zorder=1, vmin=0, vmax=1, cmap='viridis')
    if state is not None:
        plot_rope_configuration(ax, state, c='red', label='state', zorder=2)
    if next_state is not None:
        plot_rope_configuration(ax, next_state, c='orange', label='next state', zorder=3)

    plot_rope_configuration(ax, planned_state, c='blue', label='planned state', zorder=4)
    plot_rope_configuration(ax, planned_next_state, c='cyan', label='planned next state', zorder=5)

    if label is not None:
        label_color = 'g' if label else 'r'
        plt.plot([-5, 5, 5, -5, -5], [-5, -5, 5, 5, -5], c=label_color, linewidth=6)

    plt.axis("equal")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
