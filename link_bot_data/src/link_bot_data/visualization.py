import numpy as np

import rospy
from geometry_msgs.msg import Point
from link_bot_pycommon.pycommon import vector_to_points_2d
from visualization_msgs.msg import Marker


def plot_rope_configuration(ax, rope_configuration, linewidth=None, linestyle=None, s=1, label=None, scatt=True, **kwargs):
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

    arrow.scale.x = 0.005
    arrow.scale.y = 0.01
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
