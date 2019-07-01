from __future__ import division

import numpy as np


def yaw_diff(a, b):
    diff = a - b
    greater_indeces = np.argwhere(diff > np.pi)
    diff[greater_indeces] = diff[greater_indeces] - 2 * np.pi
    less_indeces = np.argwhere(diff < -np.pi)
    diff[less_indeces] = diff[less_indeces] + 2 * np.pi
    return diff


def state_cost(s, goal):
    return np.linalg.norm(s[0, 0:2] - goal[0, 0:2])


def make_random_rope_configuration(extent, length=0.5):
    while True:
        theta_1 = np.random.uniform(-np.pi, np.pi)
        theta_2 = np.random.uniform(-np.pi, np.pi)
        # don't put the head so close to the edge that the tail could be off the map
        head_x = np.random.uniform(extent[0] + 2.0 * length, extent[1] - 2.0 * length)
        head_y = np.random.uniform(extent[2] + 2.0 * length, extent[3] - 2.0 * length)

        rope_configuration = np.zeros(6)
        rope_configuration[4] = head_x
        rope_configuration[5] = head_y
        rope_configuration[2] = rope_configuration[4] + np.cos(theta_1) * length
        rope_configuration[3] = rope_configuration[5] + np.sin(theta_1) * length
        rope_configuration[0] = rope_configuration[2] + np.cos(theta_2) * length
        rope_configuration[1] = rope_configuration[3] + np.sin(theta_2) * length

        if extent[0] < rope_configuration[0] < extent[1] and \
                extent[0] < rope_configuration[2] < extent[1] and \
                extent[0] < rope_configuration[4] < extent[1] and \
                extent[2] < rope_configuration[1] < extent[3] and \
                extent[2] < rope_configuration[3] < extent[3] and \
                extent[2] < rope_configuration[5] < extent[3]:
            break

    return rope_configuration


def make_rope_configuration(head_x, head_y, theta_1, theta_2, l=0.5):
    rope_configuration = np.zeros(6)
    rope_configuration[4] = head_x
    rope_configuration[5] = head_y
    rope_configuration[2] = rope_configuration[4] + np.cos(theta_1) * l
    rope_configuration[3] = rope_configuration[5] + np.sin(theta_1) * l
    rope_configuration[0] = rope_configuration[2] + np.cos(theta_2) * l
    rope_configuration[1] = rope_configuration[3] + np.sin(theta_2) * l
    return rope_configuration


def make_rope_configurations(head_xs, head_ys, theta_1s, theta_2s, l=0.5):
    n = head_xs.shape[0]
    rope_configurations = np.zeros((n, 6))
    rope_configurations[:, 4] = head_xs
    rope_configurations[:, 5] = head_ys
    rope_configurations[:, 2] = rope_configurations[:, 4] + np.cos(theta_1s) * l
    rope_configurations[:, 3] = rope_configurations[:, 5] + np.sin(theta_1s) * l
    rope_configurations[:, 0] = rope_configurations[:, 2] + np.cos(theta_2s) * l
    rope_configurations[:, 1] = rope_configurations[:, 3] + np.sin(theta_2s) * l
    return rope_configurations
