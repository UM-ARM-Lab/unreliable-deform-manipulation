import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from ompl import base as ob

from link_bot_planning.gp_directed_control_sampler import GPDirectedControlSampler
from link_bot_planning.state_spaces import to_numpy


def flatten_plan(np_controls, np_duration_steps_int):
    # flatten and combine np_controls and durations
    np_controls_flat = []
    for control, duration in zip(np_controls, np_duration_steps_int):
        for i in range(duration):
            np_controls_flat.append(control)
    np_controls_flat = np.array(np_controls_flat)
    return np_controls_flat


def flatten_plan_dt(np_controls, np_durations, dt):
    # flatten and combine np_controls and durations
    np_duration_steps_int = (np_durations / dt).astype(np.int)
    np_controls_flat = []
    for control, duration in zip(np_controls, np_duration_steps_int):
        for i in range(duration):
            np_controls_flat.append(control)
    np_controls_flat = np.array(np_controls_flat)
    return np_controls_flat


def plot(planner_data, sdf, start, goal, path, controls, n_state, extent):
    plt.figure()
    plt.imshow(np.flipud(sdf) > 0, extent=extent)

    for state_sampled_at in GPDirectedControlSampler.states_sampled_at:
        xs = [state_sampled_at[0, 0], state_sampled_at[0, 2], state_sampled_at[0, 4]]
        ys = [state_sampled_at[0, 1], state_sampled_at[0, 3], state_sampled_at[0, 5]]
        plt.plot(xs, ys, label='sampled states', linewidth=0.5, c='b', alpha=0.5, zorder=1)

    plt.scatter(start[0, 0], start[0, 1], label='start', s=100, c='r', zorder=1)
    plt.scatter(goal[0], goal[1], label='goal', s=100, c='g', zorder=1)
    for path_i in path:
        xs = [path_i[0], path_i[2], path_i[4]]
        ys = [path_i[1], path_i[3], path_i[5]]
        plt.plot(xs, ys, label='final path', linewidth=2, c='cyan', alpha=0.75, zorder=4)
    plt.quiver(path[:-1, 4], path[:-1, 5], controls[:, 0], controls[:, 1], width=0.002, zorder=5, color='k')

    for vertex_index in range(planner_data.numVertices()):
        v = planner_data.getVertex(vertex_index)
        # draw the configuration of the rope
        s = v.getState()
        edges_map = ob.mapUintToPlannerDataEdge()

        np_s = to_numpy(s, n_state)
        plt.scatter(np_s[0, 0], np_s[0, 1], s=15, c='orange', zorder=2, alpha=0.5, label='tail')

        if len(edges_map.keys()) == 0:
            xs = [np_s[0, 0], np_s[0, 2], np_s[0, 4]]
            ys = [np_s[0, 1], np_s[0, 3], np_s[0, 5]]
            plt.plot(xs, ys, linewidth=1, c='orange', alpha=0.2, zorder=2, label='full rope')

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

    plt.legend(custom_lines, ['sampled rope configurations', 'start', 'goal', 'final path', 'controls', 'full rope', 'search tree'])
    plt.show()