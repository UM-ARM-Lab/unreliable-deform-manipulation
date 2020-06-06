from typing import Optional, List, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from ompl import base as ob

from link_bot_planning.state_spaces import compound_to_numpy
from link_bot_planning.viz_object import VizObject
from link_bot_pycommon.animation_player import Player
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from moonshine.moonshine_utils import states_are_equal, listify


def plot_plan(ax,
              state_space_description: Dict,
              scenario: ExperimentScenario,
              viz_object: VizObject,
              planner_data: ob.PlannerData,
              environment: Dict,
              goal,
              planned_path: Optional[List[Dict]],
              planned_actions: Optional[np.ndarray],
              draw_tree: Optional[bool] = None,
              draw_rejected: Optional[bool] = None,
              ):
    scenario.plot_environment(ax, environment)

    if draw_rejected:
        for rejected_state in viz_object.rejected_samples:
            scenario.plot_state(ax, rejected_state, color='orange', zorder=2, s=10, label='rejected')

    if planned_path is not None:
        start = planned_path[0]
        end = planned_path[-1]
        scenario.plot_state_simple(ax, start, color='b', s=50, zorder=5, label='start')
        scenario.plot_state_simple(ax, end, color='m', s=20, zorder=6, marker='*', label='final tail planned')
        scenario.plot_goal(ax, goal, color='c', zorder=4, s=50, label='goal')
        draw_every_n = 1
        T = len(planned_path)
        colormap = cm.winter
        for t in range(0, T, draw_every_n):
            state = planned_path[t]
            for randomly_accepted_sample in viz_object.randomly_accepted_samples:
                if states_are_equal(state, randomly_accepted_sample):
                    scenario.plot_state_simple(ax, state, color='white', s=10, zorder=4, label='random accept')
            scenario.plot_state(ax, state, color=colormap(t / T), s=10, zorder=3, label='final path')
    if draw_tree:
        print("Drawing tree.")
        tree_json = planner_data_to_json(planner_data, state_space_description)
        draw_tree_from_json(ax, scenario, tree_json)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    extent = environment['full_env/extent']
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    handles = list(by_label.values())
    labels = list(by_label.keys())
    return handles, labels


def planner_data_to_json(planner_data, state_space_description):
    json = {
        'vertices': [],
        'edges': [],
    }
    for vertex_index in range(planner_data.numVertices()):
        v = planner_data.getVertex(vertex_index)
        s = v.getState()
        edges_map = ob.mapUintToPlannerDataEdge()

        np_s = compound_to_numpy(state_space_description, s)
        json['vertices'].append(listify(np_s))

        planner_data.getEdges(vertex_index, edges_map)
        for vertex_index2 in edges_map.keys():
            v2 = planner_data.getVertex(vertex_index2)
            s2 = v2.getState()
            np_s2 = compound_to_numpy(state_space_description, s2)
            # FIXME: have a "plot edge" function in the experiment scenario?
            json['edges'].append(listify({
                'from': np_s,
                'to': np_s2,
            }))
    return json


def draw_tree_from_json(ax, scenario, tree_json):
    for state in range(tree_json['vertices']):
        scenario.plot_state(ax, state, color='k', s=10, zorder=2)

    for edge in tree_json['edges']:
        s1 = edge['from']
        s2 = edge['to']
        # FIXME: have a "plot edge" function in the experiment scenario?
        ax.plot([s1['link_bot'][0], s2['link_bot'][0]], [s1['link_bot'][1], s2['link_bot'][1]], linewidth=1, c='grey')
        scenario.plot_state_simple(ax, s2, color='k')


def animate(environment: Dict,
            scenario: ExperimentScenario,
            goal: Optional = None,
            is_close: Optional = None,
            planned_actions: Optional = None,
            planned_path: Optional[List[Dict]] = None,
            actual_path: Optional[List[Dict]] = None,
            accept_probabilities: Optional[List[float]] = None,
            fps: float = 1):
    fig = plt.figure(figsize=(20, 20))
    ax = plt.gca()
    update, frames = scenario.animate_predictions_on_axes(fig=fig,
                                                          ax=ax,
                                                          environment=environment,
                                                          actions=planned_actions,
                                                          actual=actual_path,
                                                          predictions=planned_path,
                                                          labels=is_close,
                                                          example_idx=None,
                                                          accept_probabilities=accept_probabilities,
                                                          fps=fps)
    scenario.plot_goal(ax, goal, color='c', zorder=4, s=50, label='goal')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    anim = Player(fig, update, max_index=frames, interval=1000 / fps, repeat=True)
    return anim
