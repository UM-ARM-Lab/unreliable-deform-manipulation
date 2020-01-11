from __future__ import division

import numpy as np
from colorama import Fore


def yaw_diff(a, b):
    diff = a - b
    greater_indeces = np.argwhere(diff > np.pi)
    diff[greater_indeces] = diff[greater_indeces] - 2 * np.pi
    less_indeces = np.argwhere(diff < -np.pi)
    diff[less_indeces] = diff[less_indeces] + 2 * np.pi
    return diff


def state_cost(s, goal):
    return np.linalg.norm(s[0, 0:2] - goal[0, 0:2])


def wrap_angle(angles):
    """ https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap """
    return (angles + np.pi) % (2 * np.pi) - np.pi


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_from_configuration(state):
    v1 = np.array([state[4] - state[2], state[5] - state[3]])
    v2 = np.array([state[0] - state[2], state[1] - state[3]])
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def make_random_rope_configuration(extent, n_state, total_length):
    """
    First sample a head point, then sample angles for the other points
    :param extent: bounds of the environment [xmin, xmax, ymin, ymax] (meters)
    :param total_length: length of each segment of the rope (meters)
    :return:
    """

    def oob(x, y):
        return extent[0] < x < extent[1] and extent[2] < y < extent[3]

    n_links = n_state // 2
    link_length = total_length / n_links
    valid = False
    while not valid:
        head_x = np.random.uniform(extent[0], extent[1])
        head_y = np.random.uniform(extent[2], extent[3])

        rope_configuration = np.zeros(n_state)
        rope_configuration[-2] = head_x
        rope_configuration[-1] = head_y

        j = (n_links + 1) * 2 - 1
        valid = True
        for i in range(n_links - 1):
            theta = np.random.uniform(-np.pi, np.pi)
            rope_configuration[j - 2] = rope_configuration[j - 1] + np.cos(theta) * link_length
            rope_configuration[j - 3] = rope_configuration[j] + np.sin(theta) * link_length
            j = j - 2

            if oob(rope_configuration[j], rope_configuration[j-1]):
                valid = False
                break

    return rope_configuration


def make_rope_configuration(head_x, head_y, theta_1, theta_2, l=0):
    print(Fore.YELLOW + "WARNING: This function is deprecated" + Fore.RESET)
    rope_configuration = np.zeros(6)
    rope_configuration[4] = head_x
    rope_configuration[5] = head_y
    rope_configuration[2] = rope_configuration[4] + np.cos(theta_1) * l
    rope_configuration[3] = rope_configuration[5] + np.sin(theta_1) * l
    rope_configuration[0] = rope_configuration[2] + np.cos(theta_2) * l
    rope_configuration[1] = rope_configuration[3] + np.sin(theta_2) * l
    return rope_configuration


def make_rope_configurations(head_xs, head_ys, theta_1s, theta_2s, l=0.5):
    print(Fore.YELLOW + "WARNING: This function is deprecated" + Fore.RESET)
    n = head_xs.shape[0]
    rope_configurations = np.zeros((n, 6))
    rope_configurations[:, 4] = head_xs
    rope_configurations[:, 5] = head_ys
    rope_configurations[:, 2] = rope_configurations[:, 4] + np.cos(theta_1s) * l
    rope_configurations[:, 3] = rope_configurations[:, 5] + np.sin(theta_1s) * l
    rope_configurations[:, 0] = rope_configurations[:, 2] + np.cos(theta_2s) * l
    rope_configurations[:, 1] = rope_configurations[:, 3] + np.sin(theta_2s) * l
    return rope_configurations
