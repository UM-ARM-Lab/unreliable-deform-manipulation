from enum import Enum

import numpy as np
from colorama import Fore


def sdf_indeces_to_point(rowcols, resolution, origin):
    return (rowcols - origin) * resolution


def sdf_idx_to_point(row, col, resolution, origin):
    x = (col - origin[0]) * resolution[0]
    y = (row - origin[1]) * resolution[1]
    return np.array([y, x])


def sdf_bounds(sdf, resolution, origin):
    xmin, ymin = sdf_idx_to_point(0, 0, resolution, origin)
    xmax, ymax = sdf_idx_to_point(sdf.shape[0], sdf.shape[1], resolution, origin)
    return [xmin, xmax, ymin, ymax]


def point_to_sdf_idx(x, y, resolution, origin):
    row = int(x / resolution[0] + origin[0])
    col = int(y / resolution[1] + origin[1])
    return row, col


def yaw_diff(a, b):
    diff = a - b
    greater_indeces = np.argwhere(diff > np.pi)
    diff[greater_indeces] = diff[greater_indeces] - 2 * np.pi
    less_indeces = np.argwhere(diff < -np.pi)
    diff[less_indeces] = diff[less_indeces] + 2 * np.pi
    return diff


class SDF:

    def __init__(self, sdf, gradient, resolution, origin):
        self.sdf = sdf
        self.gradient = gradient
        self.resolution = resolution
        self.origin = origin
        self.extent = sdf_bounds(sdf, resolution, origin)
        self.image = np.flipud(sdf.T)

    def save(self, sdf_filename):
        np.savez(sdf_filename,
                 sdf=self.sdf,
                 sdf_gradient=self.gradient,
                 sdf_resolution=self.resolution,
                 sdf_origin=self.origin)

    @staticmethod
    def load(filename):
        npz = np.load(filename)
        sdf = npz['sdf']
        grad = npz['sdf_gradient']
        res = npz['sdf_resolution'].reshape(2)
        origin = npz['sdf_origin'].reshape(2)
        return SDF(sdf=sdf, gradient=grad, resolution=res, origin=origin)

    def __repr__(self):
        return "SDF: size={}x{} origin=({},{}) resolution=({},{})".format(self.sdf.shape[0],
                                                                          self.sdf.shape[1],
                                                                          self.origin[0],
                                                                          self.origin[1],
                                                                          self.resolution[0],
                                                                          self.resolution[1])


def load_sdf(filename):
    npz = np.load(filename)
    sdf = npz['sdf']
    grad = npz['sdf_gradient']
    res = npz['sdf_resolution'].reshape(2)
    if 'sdf_origin' in npz:
        origin = npz['sdf_origin'].reshape(2)
    else:
        origin = np.array(sdf.shape, dtype=np.int32).reshape(2) // 2
        print(Fore.YELLOW + "WARNING: sdf npz file does not specify its origin, assume origin {}".format(origin) + Fore.RESET)
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


def make_random_rope_configuration(extent, link_length=0.5):
    theta_1 = np.random.uniform(-np.pi, np.pi)
    theta_2 = np.random.uniform(-np.pi, np.pi)
    # don't put the head so close to the edge that the tail could be off the map
    head_x = np.random.uniform(extent[0] + 2 * link_length, extent[1] - 2 * link_length)
    head_y = np.random.uniform(extent[2] + 2 * link_length, extent[3] - 2 * link_length)

    rope_configuration = np.zeros(6)
    rope_configuration[4] = head_x
    rope_configuration[5] = head_y
    rope_configuration[2] = rope_configuration[4] + np.cos(theta_1) * link_length
    rope_configuration[3] = rope_configuration[5] + np.sin(theta_1) * link_length
    rope_configuration[0] = rope_configuration[2] + np.cos(theta_2) * link_length
    rope_configuration[1] = rope_configuration[3] + np.sin(theta_2) * link_length
    return rope_configuration


def make_rope_configuration(head_x, head_y, theta_1, theta_2, link_length=0.5):
    rope_configuration = np.zeros(6)
    rope_configuration[4] = head_x
    rope_configuration[5] = head_y
    rope_configuration[2] = rope_configuration[4] + np.cos(theta_1) * link_length
    rope_configuration[3] = rope_configuration[5] + np.sin(theta_1) * link_length
    rope_configuration[0] = rope_configuration[2] + np.cos(theta_2) * link_length
    rope_configuration[1] = rope_configuration[3] + np.sin(theta_2) * link_length
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
