from typing import Iterable

import numpy as np
from geometry_msgs.msg import Point
from ompl import base as ob
from visualization_msgs.msg import MarkerArray, Marker

from link_bot_data.visualization import plot_rope_configuration
from link_bot_gazebo.gazebo_utils import GazeboServices
from link_bot_planning.state_spaces import to_numpy
from link_bot_planning.viz_object import VizObject
from link_bot_pycommon.link_bot_sdf_utils import SDF


def plot(ax,
         viz_object: VizObject,
         planner_data: ob.PlannerData,
         environment: np.ndarray,
         goal: np.ndarray,
         planned_path: np.ndarray,
         planned_actions: np.ndarray,
         extent: Iterable):
    del planned_actions
    n_state = planned_path.shape[1]
    ax.imshow(np.flipud(environment), extent=extent)

    for state_sampled_at in viz_object.states_sampled_at:
        plot_rope_configuration(ax, state_sampled_at, label='sampled states', linewidth=1.0, c='b', zorder=1)

    for rejected_state in viz_object.rejected_samples:
        plot_rope_configuration(ax, rejected_state, label='states rejected by classifier', linewidth=0.8, c='r', zorder=1)

    start = planned_path[0]
    ax.scatter(start[0], start[1], label='start', s=10, c='r', zorder=5)
    ax.scatter(goal[0], goal[1], label='goal', s=10, c='g', zorder=5)
    subsample_path_ = 2
    for rope_configuration in planned_path[::subsample_path_]:
        plot_rope_configuration(ax, rope_configuration, label='final path', linewidth=1, c='cyan', zorder=4)

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

    for vertex_index in range(planner_data.numVertices()):
        v = planner_data.getVertex(vertex_index)
        # draw the configuration of the rope
        s = v.getState()
        edges_map = ob.mapUintToPlannerDataEdge()

        # TODO: this assumes the specific compound state space I'm testing at the moment. Get the subspace by name maybe?
        np_s = to_numpy(s[0], n_state)
        ax.scatter(np_s[0, 0], np_s[0, 1], s=5, c='black', zorder=2, label='tail')

        # full rope is too noisy
        if len(edges_map.keys()) == 0:
            plot_rope_configuration(ax, np_s[0], linewidth=1, c='black', zorder=3, label='full rope')

        planner_data.getEdges(vertex_index, edges_map)
        for vertex_index2 in edges_map.keys():
            v2 = planner_data.getVertex(vertex_index2)
            s2 = v2.getState()
            np_s2 = to_numpy(s2[0], n_state)
            ax.plot([np_s[0, 0], np_s2[0, 0]], [np_s[0, 1], np_s2[0, 1]], c='gray', linewidth=0.5, zorder=1)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
    return legend


def add_sampled_configuration(services: GazeboServices,
                              np_s: np.ndarray,
                              u: np.ndarray,
                              np_s_reached: np.ndarray,
                              sdf_data: SDF):
    markers = MarkerArray()

    pre_marker = Marker()
    pre_marker.header.frame_id = '/world'
    pre_marker.scale.x = 1
    tail = Point()
    tail.x = np_s[0, 0]
    tail.y = np_s[0, 1]
    tail.z = 0
    mid = Point()
    mid.x = np_s[0, 2]
    mid.y = np_s[0, 3]
    mid.z = 0
    head = Point()
    head.x = np_s[0, 4]
    head.y = np_s[0, 5]
    head.z = 0
    pre_marker.points.append(tail)
    pre_marker.points.append(mid)
    pre_marker.points.append(head)
    pre_marker.type = Marker.LINE_STRIP
    pre_marker.color.r = 1.0
    pre_marker.color.g = 0.0
    pre_marker.color.b = 0.0
    pre_marker.color.a = 1.0

    post_marker = Marker()
    post_marker.header.frame_id = '/world'

    markers.markers.append(pre_marker)
    markers.markers.append(post_marker)
    services.rviz_sampled_configurations.publish(markers)
    return None
