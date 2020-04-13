from __future__ import division

import math
import random
import string
import warnings

import numpy as np
import tensorflow as tf
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


def angle_from_configuration(state):
    warnings.warn("invalid for multi link ropes", DeprecationWarning)
    v1 = np.array([state[4] - state[2], state[5] - state[3]])
    v2 = np.array([state[0] - state[2], state[1] - state[3]])
    return angle_2d(v1, v2)


def angle_2d(v1, v2):
    return np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2))


def batch_dot_tf(v1, v2):
    return tf.einsum('ij,ij->i', v1, v2)


def angle_2d_batch_tf(v1, v2):
    """
    :param v1: [batch, n]
    :param v2:  [batch, n]
    :return: [batch]
    """
    return tf.math.atan2(tf.linalg.det(tf.stack((v1, v2), axis=1)), batch_dot_tf(v1, v2))


def n_state_to_n_links(n_state: int):
    return int(n_state // 2 - 1)


def n_state_to_n_points(n_state: int):
    return int(n_state // 2)


def make_random_rope_configuration(extent, n_state, link_length, max_angle_rad, rng: np.random.RandomState):
    """
    First sample a head point, then sample angles for the other points
    :param max_angle_rad: NOTE, by sampling uniformly here we make certain assumptions about the planning task
    :param extent: bounds of the environment [xmin, xmax, ymin, ymax] (meters)
    :param link_length: length of each segment of the rope (meters)
    :return:
    """

    def oob(x, y):
        return not (extent[0] < x < extent[1] and extent[2] < y < extent[3])

    n_links = n_state_to_n_links(n_state)
    theta = rng.uniform(-np.pi, np.pi)
    valid = False
    while not valid:
        head_x = rng.uniform(extent[0], extent[1])
        head_y = rng.uniform(extent[2], extent[3])

        rope_configuration = np.zeros(n_state)
        rope_configuration[-2] = head_x
        rope_configuration[-1] = head_y

        j = n_state - 1
        valid = True
        for i in range(n_links):
            theta = theta + rng.uniform(-max_angle_rad, max_angle_rad)
            rope_configuration[j - 2] = rope_configuration[j] + np.cos(theta) * link_length
            rope_configuration[j - 3] = rope_configuration[j - 1] + np.sin(theta) * link_length

            if oob(rope_configuration[j - 2], rope_configuration[j - 3]):
                valid = False
                break

            j = j - 2

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


def flatten_points(points):
    return np.array([[p.x, p.y] for p in points]).flatten()


def flatten_named_points(points):
    return np.array([[p.point.x, p.point.y] for p in points]).flatten()


def get_head_from_named_points(named_points):
    for named_point in named_points:
        if named_point.name == 'head':
            head_point = np.array([named_point.point.x, named_point.point.y])
            return head_point
    raise ValueError("No head found in points {}".format(named_points))


def transpose_2d_lists(l):
    # https://stackoverflow.com/questions/6473679/transpose-list-of-lists
    return list(map(list, zip(*l)))


def print_dict(example):
    for k, v in example.items():
        print(k, v.shape)


def rand_str(length=16):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    quaternion = np.empty((4,), dtype=np.float64)
    if repetition:
        quaternion[i] = cj * (cs + sc)
        quaternion[j] = sj * (cc + ss)
        quaternion[k] = sj * (cs - sc)
        quaternion[3] = cj * (cc - ss)
    else:
        quaternion[i] = cj * sc - sj * cs
        quaternion[j] = cj * ss + sj * cc
        quaternion[k] = cj * cs - sj * sc
        quaternion[3] = cj * cc + sj * ss
    if parity:
        quaternion[j] *= -1

    return quaternion


def vector_to_points_2d(x):
    x_points = x.reshape(-1, 2)
    xs = x_points[:, 0]
    ys = x_points[:, 1]
    return xs, ys


def make_dict_float32(d):
    for k, s_k in d.items():
        d[k] = s_k.astype(np.float32)
    return d