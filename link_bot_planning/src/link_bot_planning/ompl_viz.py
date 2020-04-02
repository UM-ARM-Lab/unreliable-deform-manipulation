from typing import Optional, List, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from ompl import base as ob

from link_bot_planning.experiment_scenario import ExperimentScenario
from link_bot_planning.state_spaces import compound_to_numpy
from link_bot_planning.viz_object import VizObject
from moonshine.numpy_utils import states_are_equal


def plot_plan(ax,
              state_space_description: Dict,
              experiment_scenario: ExperimentScenario,
              viz_object: VizObject,
              planner_data: ob.PlannerData,
              environment: np.ndarray,
              goal,
              planned_path: Optional[List[Dict]],
              planned_actions: Optional[np.ndarray],
              extent,
              draw_tree: Optional[bool] = None,
              draw_rejected: Optional[bool] = None,
              ):
    ax.imshow(np.flipud(environment), extent=extent)

    # for state_sampled_at in viz_object.states_sampled_at:
    #     plot_rope_configuration(ax, state_sampled_at, label='sampled states', linewidth=1.0, c='b', zorder=1)

    if draw_rejected:
        for rejected_state in viz_object.rejected_samples:
            experiment_scenario.plot_state(ax, rejected_state, color='orange', zorder=2, s=10)

    if planned_path is not None:
        start = planned_path[0]
        end = planned_path[-1]
        experiment_scenario.plot_state_simple(ax, start, color='b', s=50, zorder=5)
        experiment_scenario.plot_state_simple(ax, end, color='pink', s=50, zorder=5, marker='*')
        experiment_scenario.plot_goal(ax, goal, color='c', zorder=5, s=50)
        draw_every_n = 1
        T = len(planned_path)
        colormap = cm.YlGn
        for t, state in range(0, T, draw_every_n):
            state = planned_path[t]
            for randomly_accepted_sample in viz_object.randomly_accepted_samples:
                if states_are_equal(state, randomly_accepted_sample):
                    experiment_scenario.plot_state_simple(ax, state, color='white', s=10, zorder=4)
            experiment_scenario.plot_state(ax, state, color=colormap(t / T), s=10, zorder=3)

    # Visualize Nearest Neighbor Selection (poorly...)
    # for sample in planner_data.getSamples():
    #     s = sample.getSampledState()
    #     neighbor_s = sample.getNeighbor()
    #     s_np = to_numpy(s[0], n_state)
    #     neighbor_s_np = to_numpy(neighbor_s[0], n_state)
    #     s_xs, s_ys, = plottable_rope_configuration(s_np)
    #     neighbor_s_xs, neighbor_s_ys, = plottable_rope_configuration(neighbor_s_np)
    #     ax.plot(s_xs, s_ys, c='r')
    #     ax.plot(neighbor_s_xs, neighbor_s_ys, c='white')
    #     ax.plot([s_np[0, 0], neighbor_s_np[0, 0]], [s_np[0, 1], neighbor_s_np[0, 1]], c='gray')

    if draw_tree:
        for vertex_index in range(planner_data.numVertices()):
            v = planner_data.getVertex(vertex_index)
            # draw the configuration of the rope
            s = v.getState()
            edges_map = ob.mapUintToPlannerDataEdge()

            np_s = compound_to_numpy(state_space_description, s)
            experiment_scenario.plot_state_simple(ax, np_s, color='k')

            # full rope is too noisy
            # if len(edges_map.keys()) == 0:
            #     plot_rope_configuration(ax, np_s[0], linewidth=1, c='black', zorder=3, label='full rope')

            planner_data.getEdges(vertex_index, edges_map)
            for vertex_index2 in edges_map.keys():
                v2 = planner_data.getVertex(vertex_index2)
                s2 = v2.getState()
                np_s2 = compound_to_numpy(state_space_description, s2)
                experiment_scenario.plot_state_simple(ax, np_s2, color='k')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))

    return legend


def plan_vs_execution(environment: np.ndarray,
                      extent,
                      experiment_scenario: ExperimentScenario,
                      goal,
                      planned_path: Optional[List[Dict[str, np.ndarray]]] = None,
                      actual_path: Optional[List[Dict[str, np.ndarray]]] = None):
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(np.flipud(environment), extent=extent)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])

    start = planned_path[0]
    experiment_scenario.plot_state(ax, start, color='b', zorder=2, s=20)
    experiment_scenario.plot_goal(ax, goal, color='c', zorder=2, s=20)

    if planned_path is not None:
        planned_path_artist = experiment_scenario.plot_state(ax, planned_path[0], 'g', zorder=3, s=20)
    if actual_path is not None:
        actual_path_artist = experiment_scenario.plot_state(ax, actual_path[0], '#00ff00', zorder=3, s=20)
    plt.legend()

    def update(t):
        if planned_path is not None:
            experiment_scenario.update_artist(planned_path_artist, planned_path[t])
        if actual_path is not None:
            experiment_scenario.update_artist(actual_path_artist, actual_path[t])

    anim = FuncAnimation(fig, update, frames=len(planned_path), interval=500)
    return anim
