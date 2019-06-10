import numpy as np
from enum import Enum


def sdf_indeces_to_point(rowcols, resolution, origin):
    return (rowcols - origin) * resolution


def sdf_idx_to_point(row, col, resolution, sdf_origin):
    x = (col - sdf_origin[0, 0]) * resolution[0, 0]
    y = (row - sdf_origin[1, 0]) * resolution[1, 0]
    return np.array([[y], [x]])


def point_to_sdf_idx(x, y, resolution, sdf_origin):
    row = int(x / resolution[0] + sdf_origin[0])
    col = int(y / resolution[1] + sdf_origin[1])
    return row, col


def yaw_diff(a, b):
    diff = a - b
    greater_indeces = np.argwhere(diff > np.pi)
    diff[greater_indeces] = diff[greater_indeces] - 2 * np.pi
    less_indeces = np.argwhere(diff < -np.pi)
    diff[less_indeces] = diff[less_indeces] + 2 * np.pi
    return diff


def load_sdf(filename):
    npz = np.load(filename)
    sdf = npz['sdf']
    grad = npz['sdf_gradient']
    res = npz['sdf_resolution'].reshape(2)
    origin = np.array(sdf.shape, dtype=np.int32).reshape(2) // 2
    return sdf, grad, res, origin


def state_cost(s, goal):
    return np.linalg.norm(s[0, 0:2] - goal[0, 0:2])


class ArgsEnum(Enum):

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return cls[s]
        except KeyError:
            raise ValueError()


def make_rope_configuration(head_x, head_y, theta_1, theta_2):
    rope_configuration = np.zeros(6)
    rope_configuration[4] = head_x
    rope_configuration[5] = head_y
    rope_configuration[2] = rope_configuration[4] + np.cos(theta_1)
    rope_configuration[3] = rope_configuration[5] + np.sin(theta_1)
    rope_configuration[0] = rope_configuration[2] + np.cos(theta_2)
    rope_configuration[1] = rope_configuration[3] + np.sin(theta_2)
    return rope_configuration
