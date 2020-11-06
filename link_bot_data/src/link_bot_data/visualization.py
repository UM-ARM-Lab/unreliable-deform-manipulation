from typing import Dict, Optional, Callable

import numpy as np

import rospy
from geometry_msgs.msg import Point
from link_bot_data.dataset_utils import index_time_with_metadata, add_predicted
from link_bot_pycommon.experiment_scenario import ExperimentScenario
from link_bot_pycommon.pycommon import vector_to_points_2d
from merrrt_visualization.rviz_animation_controller import RvizAnimationController, RvizAnimation
from moonshine.moonshine_utils import numpify
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker


def plot_rope_configuration(ax, rope_configuration, linewidth=None, linestyle=None, s=1, label=None, scatt=True,
                            **kwargs):
    xs, ys = vector_to_points_2d(rope_configuration)
    if scatt:
        ax.scatter(xs, ys, s=s, **kwargs)
    return ax.plot(xs, ys, linewidth=linewidth, label=label, linestyle=linestyle, **kwargs)


def my_arrow(xs, ys, us, vs, scale=0.2):
    xs = np.array(xs)
    ys = np.array(ys)
    us = np.array(us)
    vs = np.array(vs)

    thetas = np.arctan2(vs, us)
    head_lengths = np.sqrt(np.square(us) + np.square(vs)) * scale
    theta1s = 3 * np.pi / 4 + thetas
    u1_primes = np.cos(theta1s) * head_lengths
    v1_primes = np.sin(theta1s) * head_lengths
    theta2s = thetas - 3 * np.pi / 4
    u2_primes = np.cos(theta2s) * head_lengths
    v2_primes = np.sin(theta2s) * head_lengths

    return ([xs, xs + us], [ys, ys + vs]), \
           ([xs + us, xs + us + u1_primes], [ys + vs, ys + vs + v1_primes]), \
           ([xs + us, xs + us + u2_primes], [ys + vs, ys + vs + v2_primes])


def plot_arrow(ax, xs, ys, us, vs, color, **kwargs):
    xys1, xys2, xys3 = my_arrow(xs, ys, us, vs)
    lines = []
    lines.append(ax.plot(xys1[0], xys1[1], c=color, **kwargs)[0])
    lines.append(ax.plot(xys2[0], xys2[1], c=color, **kwargs)[0])
    lines.append(ax.plot(xys3[0], xys3[1], c=color, **kwargs)[0])
    return lines


def update_arrow(lines, xs, ys, us, vs):
    xys1, xys2, xys3 = my_arrow(xs, ys, us, vs)
    lines[0].set_data(xys1[0], xys1[1])
    lines[1].set_data(xys2[0], xys2[1])
    lines[2].set_data(xys3[0], xys3[1])


def plot_extents(ax, extent, linewidth=6, **kwargs):
    line = ax.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
                   [extent[2], extent[2], extent[3], extent[3], extent[2]],
                   linewidth=linewidth,
                   **kwargs)[0]
    return line


def rviz_arrow(position: np.ndarray,
               target_position: np.ndarray,
               r: float,
               g: float,
               b: float,
               a: float,
               label: str = 'arrow',
               idx: int = 0,
               **kwargs):
    arrow = Marker()
    arrow.action = Marker.ADD  # create or modify
    arrow.type = Marker.ARROW
    arrow.header.frame_id = "world"
    arrow.header.stamp = rospy.Time.now()
    arrow.ns = label
    arrow.id = idx

    arrow.scale.x = 0.01
    arrow.scale.y = 0.02
    arrow.scale.z = 0

    arrow.pose.orientation.w = 1

    start = Point()
    start.x = position[0]
    start.y = position[1]
    start.z = position[2]
    end = Point()
    end.x = target_position[0]
    end.y = target_position[1]
    end.z = target_position[2]
    arrow.points.append(start)
    arrow.points.append(end)

    arrow.color.r = r
    arrow.color.g = g
    arrow.color.b = b
    arrow.color.a = a

    return arrow


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.figure()
    ax = plt.gca()
    r = 4
    for theta in np.linspace(-np.pi, np.pi, 36):
        u = np.cos(theta) * r
        v = np.sin(theta) * r
        plot_arrow(ax, 2, 0, u, v, 'r')
    plt.axis("equal")
    plt.show()


def classifier_transition_viz_t(predicted_state_keys, true_state_keys: Optional):
    def _classifier_transition_viz_t(scenario: ExperimentScenario, example: Dict, t: int):
        pred_t = index_time_with_metadata(scenario, example, predicted_state_keys, t=t)
        scenario.plot_state_rviz(pred_t, label='predicted', color='#0000ffff')

        label_t = example['is_close'][t]
        scenario.plot_is_close(label_t)

        if true_state_keys is not None:
            true_t = index_time_with_metadata(scenario, example, true_state_keys, t=t)
            scenario.plot_state_rviz(true_t, label='actual', color='#ff0000ff', scale=1.1)

    return _classifier_transition_viz_t


def init_viz_action(action_keys, state_keys):
    def _init_viz_action(scenario: ExperimentScenario, example: Dict):
        action = {k: example[k][0] for k in action_keys}
        pred_0 = index_time_with_metadata(scenario, example, state_keys, t=0)
        scenario.plot_action_rviz(pred_0, action)

    return _init_viz_action


def init_viz_env(scenario: ExperimentScenario, example: Dict):
    scenario.plot_environment_rviz(example)


def stdev_viz_t(pub: rospy.Publisher):
    def _stdev_viz_t(scenario: ExperimentScenario, example: Dict, t: int):
        stdev_t = example[add_predicted('stdev')][t, 0]
        stdev_msg = Float32()
        stdev_msg.data = stdev_t
        pub.publish(stdev_msg)

    return _stdev_viz_t
