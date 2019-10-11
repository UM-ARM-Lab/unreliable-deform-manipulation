import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from ompl import base as ob

from link_bot_data.visualization import plot_rope_configuration
from link_bot_planning.state_spaces import to_numpy


def plot(sampler, planner_data, sdf, goal, planned_path, planned_actions, extent):
    plt.figure()
    ax = plt.gca()
    n_state = planned_path.shape[1]
    plt.imshow(np.flipud(sdf) > 0, extent=extent)

    for state_sampled_at in sampler.states_sampled_at:
        xs = [state_sampled_at[0, 0], state_sampled_at[0, 2], state_sampled_at[0, 4]]
        ys = [state_sampled_at[0, 1], state_sampled_at[0, 3], state_sampled_at[0, 5]]
        plt.plot(xs, ys, label='sampled states', linewidth=0.5, c='b', alpha=0.5, zorder=1)

    start = planned_path[0]
    plt.scatter(start[0], start[1], label='start', s=100, c='r', zorder=1)
    plt.scatter(goal[0], goal[1], label='goal', s=100, c='g', zorder=1)
    for rope_configuration in planned_path:
        plot_rope_configuration(ax, rope_configuration, label='final path', linewidth=2, c='cyan', alpha=0.75, zorder=4)

    # Visualize actions
    # the -1 excludes the final configuration, where there is no corresponding action
    plt.quiver(planned_path[:-1, 4], planned_path[:-1, 5], planned_actions[:, 0], planned_actions[:, 1],
               width=0.001, zorder=2, color='k', alpha=0.2)

    for vertex_index in range(planner_data.numVertices()):
        v = planner_data.getVertex(vertex_index)
        # draw the configuration of the rope
        s = v.getState()
        edges_map = ob.mapUintToPlannerDataEdge()

        np_s = to_numpy(s, n_state)
        plt.scatter(np_s[0, 0], np_s[0, 1], s=15, c='orange', zorder=2, alpha=0.5, label='tail')

        if len(edges_map.keys()) == 0:
            plot_rope_configuration(ax, np_s[0], linewidth=1, c='orange', alpha=0.2, zorder=2, label='full rope')

        planner_data.getEdges(vertex_index, edges_map)
        for vertex_index2 in edges_map.keys():
            v2 = planner_data.getVertex(vertex_index2)
            s2 = v2.getState()
            np_s2 = to_numpy(s2, n_state)
            plt.plot([np_s[0, 0], np_s2[0, 0]], [np_s[0, 1], np_s2[0, 1]], c='white', label='tree', zorder=1)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.xlim(extent[0:2])
    plt.ylim(extent[2:4])

    custom_lines = [
        Line2D([0], [0], color='b', lw=1),
        Line2D([0], [0], color='r', lw=1),
        Line2D([0], [0], color='g', lw=1),
        Line2D([0], [0], color='cyan', lw=1),
        Line2D([0], [0], color='k', lw=1),
        Line2D([0], [0], color='orange', lw=1),
        Line2D([0], [0], color='orange', lw=1),
        Line2D([0], [0], color='white', lw=1),
    ]

    plt.legend(custom_lines,
               ['sampled rope configurations', 'start', 'goal', 'final path', 'controls', 'full rope', 'search tree'])
    plt.show()
